from __future__ import annotations

import math
import numpy as np
import cv2
import mediapipe as mp

from core.config import (
    TARGET_SIZE, TINY_FACE, SMALL_FACE,
    NLM_H_COLOR, NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
    IBP_ITERATIONS, IBP_STEP_GAIN,
    UNSHARP_INTERMEDIATE, UNSHARP_ZONE_HOT, UNSHARP_ZONE_COOL, UNSHARP_FALLBACK,
    MP_MIN_DETECTION_CONF, MP_REFINE_LANDMARKS,
    MASK_FEATHER_KERNEL, ALIGN_MAX_ANGLE_DEG,
    HOT_ZONE, LEFT_EYE_CENTRE_IDX, RIGHT_EYE_CENTRE_IDX,
    LEFT_EYE_FALLBACK, RIGHT_EYE_FALLBACK,
)


# MediaPipe singleton — loading this once saves ~6 seconds across 100 faces
_face_mesh = None


def _get_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=MP_REFINE_LANDMARKS,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
        )
    return _face_mesh


def cleanup_mesh():
    global _face_mesh
    if _face_mesh is not None:
        _face_mesh.close()
        _face_mesh = None


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------

def _noise_sigma(img: np.ndarray) -> float:
    # Donoho-Johnstone MAD estimator adapted to spatial Laplacian.
    # More reliable than std on the raw image because it ignores edges.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap  = cv2.Laplacian(gray, cv2.CV_32F)
    return max(float(np.median(np.abs(lap))) / 0.6745, 1.0)


# ---------------------------------------------------------------------------
# Stage 1 — Denoising
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(h, w)

    # For very small faces NLM averages over too few pixels and smears
    # the little structure that exists.  Bilateral filter preserves edges
    # much better at this scale.
    if short <= TINY_FACE:
        return cv2.bilateralFilter(img, d=5, sigmaColor=20, sigmaSpace=5)

    sigma = _noise_sigma(img)
    h_val = int(np.clip(sigma * 0.4, 4, 12))

    return cv2.fastNlMeansDenoisingColored(
        img, None,
        h=h_val,
        hColor=NLM_H_COLOR,
        templateWindowSize=NLM_TEMPLATE_WINDOW,
        searchWindowSize=NLM_SEARCH_WINDOW,
    )


# ---------------------------------------------------------------------------
# Stage 2 — CLAHE
# ---------------------------------------------------------------------------

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    # CLAHE tiles need at least a few pixels each to be meaningful.
    # On a 12 px face with tileGridSize=(4,4) each tile is 3×3 — useless.
    # Scale the tile grid down proportionally so tiles stay ≥ 8 px wide.
    tile = max(1, min(h, w) // 8)
    tile_grid = (tile, tile)

    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=tile_grid)
    l_enh   = clahe.apply(l)

    return cv2.cvtColor(cv2.merge([l_enh, a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Unsharp mask — shared by stages 3 and 4
# ---------------------------------------------------------------------------

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    blurred   = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# IBP — Iterative Back-Projection super-resolution
#
# The idea: upscale the image, then iteratively simulate the degradation
# (downscale back to original size), compare with the actual original,
# upscale the residual and add it back.  Each iteration drives the
# high-resolution estimate closer to one that is consistent with the
# observed low-resolution input.
#
# This is exactly the classical algorithm from Irani & Peleg (1991) and it
# works dramatically better than a single-step LANCZOS for < 64 px faces.
# ---------------------------------------------------------------------------

def _ibp_upscale(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    iterations: int = IBP_ITERATIONS,
    gain: float = IBP_STEP_GAIN,
) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    src_f  = img.astype(np.float32)

    # Warm start: LANCZOS4 gives a better initial estimate than bicubic
    current = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

    for _ in range(iterations):
        # Simulate the camera blur + downscale (the degradation model)
        sim_low = cv2.resize(current, (src_w, src_h), interpolation=cv2.INTER_AREA)

        # Where does our current estimate deviate from the real observation?
        residual = src_f - sim_low

        # Project the residual correction back to the high-res domain
        residual_up = cv2.resize(residual, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Damped update prevents oscillation
        current = np.clip(current + gain * residual_up, 0, 255)

    return current.astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 3 — Upscaling
# ---------------------------------------------------------------------------

def stage3_upscale(img: np.ndarray) -> np.ndarray:
    h, w    = img.shape[:2]
    short   = min(h, w)
    tw, th  = TARGET_SIZE

    if short <= TINY_FACE:
        # Tiny path: double twice with IBP refinement at each step so we
        # never ask for more than a ~4x jump in a single pass.
        step1_w, step1_h = w * 2, h * 2
        img = _ibp_upscale(img, step1_w, step1_h)
        img = unsharp_mask(img, *UNSHARP_INTERMEDIATE)

        step2_w, step2_h = min(step1_w * 2, tw * 2), min(step1_h * 2, th * 2)
        img = _ibp_upscale(img, step2_w, step2_h)
        img = unsharp_mask(img, *UNSHARP_INTERMEDIATE)

        # Final LANCZOS resize to exact target
        return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    if short <= SMALL_FACE:
        # Small path: two LANCZOS passes with IBP refinement between them
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        img = unsharp_mask(img, *UNSHARP_INTERMEDIATE)
        h2, w2 = img.shape[:2]
        img = _ibp_upscale(img, w2 * 2, h2 * 2)
        return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    # Already decent size — LANCZOS4 direct resize is fine
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# Stage 4 helpers — alignment and mask
# ---------------------------------------------------------------------------

def _lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def _eye_centre(landmarks, indices, fallback, w, h):
    pts = []
    for i in indices:
        if i < len(landmarks):
            pts.append(_lm_px(landmarks[i], w, h))
    if not pts:
        for i in fallback:
            if i < len(landmarks):
                pts.append(_lm_px(landmarks[i], w, h))
    return np.mean(pts, axis=0).astype(np.float32) if pts else None


def _align(img: np.ndarray, landmarks) -> np.ndarray:
    h, w = img.shape[:2]
    lc = _eye_centre(landmarks, [LEFT_EYE_CENTRE_IDX],  LEFT_EYE_FALLBACK,  w, h)
    rc = _eye_centre(landmarks, [RIGHT_EYE_CENTRE_IDX], RIGHT_EYE_FALLBACK, w, h)
    if lc is None or rc is None:
        return img

    angle = math.degrees(math.atan2(float(rc[1] - lc[1]), float(rc[0] - lc[0])))
    if abs(angle) > ALIGN_MAX_ANGLE_DEG:
        return img

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT_101)


def _build_mask(landmarks, h, w) -> np.ndarray:
    pts = [_lm_px(landmarks[i], w, h) for i in HOT_ZONE if i < len(landmarks)]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        cv2.fillPoly(mask, [hull], 255)
    k = MASK_FEATHER_KERNEL | 1
    soft = cv2.GaussianBlur(mask, (k, k), 0)
    return soft.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Laplacian pyramid blend — no visible zone boundary
# ---------------------------------------------------------------------------

def _pyr_blend(strong: np.ndarray, gentle: np.ndarray, mask: np.ndarray, levels: int = 4) -> np.ndarray:
    def gauss_pyr(img):
        g = [img.astype(np.float32)]
        for _ in range(levels):
            g.append(cv2.pyrDown(g[-1]))
        return g

    def lap_pyr(gp):
        lp = []
        for i in range(len(gp) - 1):
            up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1].copy())
        return lp

    def collapse(lp):
        img = lp[-1]
        for layer in reversed(lp[:-1]):
            img = cv2.pyrUp(img, dstsize=(layer.shape[1], layer.shape[0])) + layer
        return img

    mask3 = np.stack([mask] * 3, axis=-1)
    gm    = gauss_pyr(mask3)

    ls = lap_pyr(gauss_pyr(strong.astype(np.float32)))
    lg = lap_pyr(gauss_pyr(gentle.astype(np.float32)))

    blended = [ls[i] * gm[i] + lg[i] * (1.0 - gm[i]) for i in range(len(ls))]
    return np.clip(collapse(blended), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 4 — Zone sharpening
# ---------------------------------------------------------------------------

def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    mesh = _get_mesh()
    h, w = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res  = mesh.process(rgb)

    if not res.multi_face_landmarks:
        s, st = UNSHARP_FALLBACK
        return unsharp_mask(img, s, st)

    lm = res.multi_face_landmarks[0].landmark

    # Align face so eyes are horizontal — improves recognition accuracy
    aligned = _align(img, lm)

    # Re-detect on aligned image for an accurate mask
    res2 = mesh.process(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    lm2  = res2.multi_face_landmarks[0].landmark if res2.multi_face_landmarks else lm

    zone_mask = _build_mask(lm2, h, w)

    strong = unsharp_mask(aligned, *UNSHARP_ZONE_HOT)
    gentle = unsharp_mask(aligned, *UNSHARP_ZONE_COOL)

    return _pyr_blend(strong, gentle, zone_mask, levels=4)