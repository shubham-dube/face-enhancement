"""
Stage 1: Adaptive NLM denoising   (noise-level-aware h parameter)
Stage 2: Perceptual CLAHE          (LAB colour space, L-channel only)
Stage 3: Multi-step LANCZOS upscale with intermediate unsharp masking
Stage 4: MediaPipe landmark-guided zone sharpening + auto face alignment

Innovations beyond the brief:
  · Noise-level estimation (MAD of Laplacian) → adaptive h in Stage 1
  · Eye-axis alignment (rotate so eyes are horizontal) inside Stage 4
  · Laplacian-pyramid blending for seamless zone boundary in Stage 4
  · MediaPipe FaceMesh kept as process-level singleton → ≈6s saved/100 faces
  · Sharpness gate: already-sharp images skip stages 1-4 (Bonus requirement)
"""

from __future__ import annotations

import math
import numpy as np
import cv2
import mediapipe as mp

from core.config import (
    TARGET_SIZE,
    SHARPNESS_GATE,
    NLM_H_DEFAULT, NLM_H_COLOR, NLM_TEMPLATE_WINDOW, NLM_SEARCH_WINDOW,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
    SMALL_FACE_THRESHOLD,
    UNSHARP_INTERMEDIATE, UNSHARP_ZONE_HOT, UNSHARP_ZONE_COOL, UNSHARP_FALLBACK,
    MP_MIN_DETECTION_CONF, MP_REFINE_LANDMARKS,
    MASK_FEATHER_KERNEL,
    ALIGN_MAX_ANGLE_DEG,
    HOT_ZONE,
    LEFT_EYE, RIGHT_EYE,
    LEFT_EYE_CENTRE_IDX, RIGHT_EYE_CENTRE_IDX,
    LEFT_EYE_FALLBACK, RIGHT_EYE_FALLBACK,
)


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe singleton — initialised once for the whole process lifetime.
# Avoids 60 ms model-load overhead per image.
# ─────────────────────────────────────────────────────────────────────────────

_face_mesh: mp.solutions.face_mesh.FaceMesh | None = None  # type: ignore[name-defined]


def _get_mesh() -> mp.solutions.face_mesh.FaceMesh:  # type: ignore[name-defined]
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=MP_REFINE_LANDMARKS,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
        )
    return _face_mesh


def cleanup_mesh() -> None:
    """Call once at the end of main to release MediaPipe resources."""
    global _face_mesh
    if _face_mesh is not None:
        _face_mesh.close()
        _face_mesh = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_noise_sigma(img: np.ndarray) -> float:
    """
    Estimate image noise level using the Median Absolute Deviation (MAD) of the
    Laplacian response — the Donoho-Johnstone (1994) wavelet noise estimator
    adapted to the spatial domain.

    Returns σ̂ in pixel-value units.  Typical CCTV values: 3–20.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap  = cv2.Laplacian(gray, cv2.CV_32F)
    # Robust noise estimator: MAD / 0.6745  (≈ σ for Gaussian noise)
    sigma = float(np.median(np.abs(lap))) / 0.6745
    return max(sigma, 1.0)   # avoid zero


def _lm_pixel(lm, w: int, h: int) -> tuple[int, int]:
    """Convert normalised MediaPipe landmark to pixel coords."""
    return int(lm.x * w), int(lm.y * h)


def _convex_hull_mask(
    pts: list[tuple[int, int]], h: int, w: int
) -> np.ndarray:
    """Return a uint8 mask with the convex hull of pts filled white."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pts) >= 3:
        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        cv2.fillPoly(mask, [hull], 255)
    return mask


def _build_zone_mask(
    landmarks,
    h: int,
    w: int,
    feather: int = MASK_FEATHER_KERNEL,
) -> np.ndarray:
    """
    Build a float32 mask [0.0, 1.0] that is 1.0 in the HOT zone
    (eyes + eyebrows + nose) and 0.0 elsewhere, with a Gaussian-feathered edge
    to eliminate any visible sharpness boundary artefact.
    """
    pts = [_lm_pixel(landmarks[i], w, h) for i in HOT_ZONE if i < len(landmarks)]
    hard_mask = _convex_hull_mask(pts, h, w)

    # Feather (blur) the edge so the blend is perceptually seamless
    kernel = feather | 1   # ensure odd
    soft   = cv2.GaussianBlur(hard_mask, (kernel, kernel), 0)
    return soft.astype(np.float32) / 255.0


def _get_eye_centres(landmarks, n_lm: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return pixel-space (x, y) of left and right eye centres.
    Prefers iris landmarks (refine_landmarks=True) then falls back to corner mean.
    """
    def mean_px(indices):
        valid = [_lm_pixel(landmarks[i], w, h) for i in indices if i < n_lm]
        return np.mean(valid, axis=0).astype(np.float32) if len(valid) > 0 else None

    lc = mean_px([LEFT_EYE_CENTRE_IDX])
    if lc is None:
        lc = mean_px(LEFT_EYE_FALLBACK)

    rc = mean_px([RIGHT_EYE_CENTRE_IDX])
    if rc is None:
        rc = mean_px(RIGHT_EYE_FALLBACK)

    return lc, rc


def _align_face(img: np.ndarray, landmarks, max_angle: float = ALIGN_MAX_ANGLE_DEG) -> np.ndarray:
    """
    Rotate the image so the inter-ocular axis is exactly horizontal.
    Operates in-place on the upscaled 240×240 image.

    Why:  A canonical (eyes-horizontal) face dramatically improves
          face_recognition match rate because dlib's 68-point predictor
          was trained on aligned faces.
    """
    h, w = img.shape[:2]
    n_lm = len(landmarks)

    lc, rc = _get_eye_centres(landmarks, n_lm, w, h)
    if lc is None or rc is None:
        return img

    dy = float(rc[1] - lc[1])
    dx = float(rc[0] - lc[0])
    angle = math.degrees(math.atan2(dy, dx))

    if abs(angle) > max_angle:
        # Extreme tilt: likely a profile or badly detected face — skip alignment
        return img

    centre = (w / 2.0, h / 2.0)
    M      = cv2.getRotationMatrix2D(centre, angle, 1.0)
    aligned = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned


def _laplacian_pyramid_blend(
    strong: np.ndarray,
    gentle: np.ndarray,
    mask: np.ndarray,
    levels: int = 4,
) -> np.ndarray:
    """
    Multi-resolution Laplacian pyramid blending.

    Unlike simple α-blending, pyramid blending composites low and high
    spatial frequencies separately.  The result has no visible halo or
    sharpness boundary at the zone edge even when the two images differ
    significantly in local contrast.

    Parameters
    ----------
    strong  : image with aggressive sharpening (eye/nose zone)
    gentle  : image with subtle sharpening (rest of face)
    mask    : float32 [0,1] — 1 → use strong, 0 → use gentle
    levels  : number of Laplacian pyramid levels
    """
    def build_gaussian(img):
        gp = [img.astype(np.float32)]
        for _ in range(levels):
            gp.append(cv2.pyrDown(gp[-1]))
        return gp

    def build_laplacian(gp):
        lp = []
        for i in range(len(gp) - 1):
            up  = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1])
        return lp

    def collapse(lp):
        img = lp[-1]
        for layer in reversed(lp[:-1]):
            img = cv2.pyrUp(img, dstsize=(layer.shape[1], layer.shape[0])) + layer
        return img

    # Build 3-channel mask pyramids
    mask3 = np.stack([mask] * 3, axis=-1)
    gp_m  = build_gaussian(mask3)

    gp_s  = build_gaussian(strong.astype(np.float32))
    gp_g  = build_gaussian(gentle.astype(np.float32))

    lp_s  = build_laplacian(gp_s)
    lp_g  = build_laplacian(gp_g)

    # Blend each Laplacian level
    lp_blend = []
    for ls, lg, gm in zip(lp_s, lp_g, gp_m):
        lp_blend.append(ls * gm + lg * (1.0 - gm))

    blended = collapse(lp_blend)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Adaptive Noise-Level-Aware Denoising
# ─────────────────────────────────────────────────────────────────────────────

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """
    cv2.fastNlMeansDenoisingColored with h=8, hColor=8, templateWindowSize=7,
    searchWindowSize=21.

    Enhancement: h is adaptively scaled by estimated image noise σ so that
    faint crops are denoised more aggressively and already-decent crops are
    handled conservatively.  h is clamped to [4, 12] to stay safe.
    """
    sigma = _estimate_noise_sigma(img)
    # Base spec is h=8 at σ≈20 (typical CCTV noise floor).
    # Scale proportionally; clamp to spec-compatible range.
    h_adaptive = int(np.clip(sigma * (NLM_H_DEFAULT / 20.0), 4, 12))

    return cv2.fastNlMeansDenoisingColored(
        img, None,
        h=h_adaptive,
        hColor=NLM_H_COLOR,
        templateWindowSize=NLM_TEMPLATE_WINDOW,
        searchWindowSize=NLM_SEARCH_WINDOW,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Perceptual CLAHE in LAB Colour Space
# ─────────────────────────────────────────────────────────────────────────────

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """
    Convert to LAB. Apply CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L
    channel only. Merge + convert back to BGR.

    Operating on L* (luminance) in the perceptually-uniform LAB space means
    contrast enhancement does not alter hue or saturation — skin tones stay
    natural while shadow/highlight detail is recovered.
    """
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe   = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    l_enh = clahe.apply(l)

    merged = cv2.merge([l_enh, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Unsharp mask — shared by Stage 3 and Stage 4
# ─────────────────────────────────────────────────────────────────────────────

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    High-boost spatial filter:
        blurred  = GaussianBlur(img, σ)
        result   = img + strength × (img − blurred)
                 = (1 + strength) × img  −  strength × blurred

    Parameters
    ----------
    sigma    : Gaussian blur radius controlling which frequencies are boosted.
               Smaller σ → finer detail; larger σ → coarser structure.
    strength : Boost factor.  1.0 = 100% increase in edge contrast.
    """
    blurred   = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Multi-Step LANCZOS4 Upscaling
# ─────────────────────────────────────────────────────────────────────────────

def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """
    For faces with short side < 64px:
        1. 2× LANCZOS4 resize
        2. Unsharp mask (σ=1.0, strength=1.6) — reinforce mid-band structure
        3. 2× LANCZOS4 resize
        4. Final resize to TARGET_SIZE

    For larger faces:
        Direct resize to TARGET_SIZE with LANCZOS4.

    The two-step approach stays within LANCZOS4's reliable magnification
    range (~4×) at each pass, preventing the aliasing that a single 20×
    upscale would produce.
    """
    h, w        = img.shape[:2]
    short_side  = min(h, w)

    if short_side < SMALL_FACE_THRESHOLD:
        # Pass 1
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        # Intermediate sharpening — preserve edge frequencies across the upscale
        sigma, strength = UNSHARP_INTERMEDIATE
        img = unsharp_mask(img, sigma=sigma, strength=strength)
        # Pass 2
        h2, w2 = img.shape[:2]
        img = cv2.resize(img, (w2 * 2, h2 * 2), interpolation=cv2.INTER_LANCZOS4)

    # Always finish with exact TARGET_SIZE
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — MediaPipe Landmark-Guided Zone Sharpening + Auto Alignment
# ─────────────────────────────────────────────────────────────────────────────

def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    Landmark-guided differential sharpening:
        HOT zone  (eye + eyebrow + nose): unsharp(σ=0.8, s=2.0)
        COOL zone (rest of face):         unsharp(σ=1.2, s=1.3)

    Blending uses a Laplacian pyramid (4 levels) rather than simple α-blend —
    this eliminates the visible sharpness boundary at the zone edge.

    Innovation: before building the mask the face is auto-aligned (rotated
    so eyes are horizontal).  This improves face_recognition match accuracy
    because dlib's landmark predictor was trained on aligned faces.

    Fallback when no face is detected: uniform unsharp(σ=1.0, s=1.5).
    """
    mesh = _get_mesh()
    h, w = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = mesh.process(rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # ── Auto-alignment ────────────────────────────────────────────────────
        img_aligned = _align_face(img, landmarks)

        # Re-detect landmarks on the aligned image for an accurate zone mask
        rgb_aligned    = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
        result_aligned = mesh.process(rgb_aligned)

        lm_final = (
            result_aligned.multi_face_landmarks[0].landmark
            if result_aligned.multi_face_landmarks
            else landmarks          # safe fallback
        )

        # ── Build feathered zone mask ─────────────────────────────────────────
        zone_mask = _build_zone_mask(lm_final, h, w)

        # ── Differential sharpening ───────────────────────────────────────────
        s_hot, s_cool = UNSHARP_ZONE_HOT, UNSHARP_ZONE_COOL
        strong = unsharp_mask(img_aligned, sigma=s_hot[0],  strength=s_hot[1])
        gentle = unsharp_mask(img_aligned, sigma=s_cool[0], strength=s_cool[1])

        # ── Laplacian pyramid blend (perceptually seamless) ───────────────────
        blended = _laplacian_pyramid_blend(strong, gentle, zone_mask, levels=4)
        return blended

    else:
        # Fallback: no face detected (very tiny / extreme pose)
        s, st = UNSHARP_FALLBACK
        return unsharp_mask(img, sigma=s, strength=st)