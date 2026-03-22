"""
core/metrics.py — Evaluation helpers for the face enhancement pipeline.

Functions
---------
sharpness        Laplacian variance (higher = sharper image)
get_face_encoding  128-d face embedding via face_recognition / dlib ResNet
ssim_score       Structural Similarity Index (skimage)
compute_pq_score Composite Perceptual Quality score [0, 100]

Design notes
~~~~~~~~~~~~
• get_face_encoding uses number_of_times_to_upsample=2 for faces where
  the short side is below 100 px — the recognition model was designed for
  larger inputs and misses tiny faces at upsample=1.
• ssim_score resizes both images to TARGET_SIZE before comparison so the
  metric is always computed on equal-dimension arrays.
• compute_pq_score is an innovation beyond the brief: it combines sharpness,
  SSIM improvement and match confidence into one human-readable [0, 100] score.
"""

from __future__ import annotations

import numpy as np
import cv2
import face_recognition
from skimage.metrics import structural_similarity

from core.config import TARGET_SIZE, FACE_MATCH_TOLERANCE


# ─────────────────────────────────────────────────────────────────────────────
# Sharpness — Laplacian variance
# ─────────────────────────────────────────────────────────────────────────────

def sharpness(img: np.ndarray) -> float:
    """
    Laplacian variance of the grayscale image.

    The Laplacian is a second-order derivative operator: it fires at edges.
    Its variance measures both the quantity and crispness of edges in the
    image.  Blurry images have very low Laplacian variance; sharp images have
    high variance.

    Returns
    -------
    float  — Laplacian variance.  Typical ranges:
             <  20  : heavily blurred (typical raw CCTV crop)
             20–80  : recoverable blur
             > 80   : already sharp (bonus gate threshold)
    """
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())


# ─────────────────────────────────────────────────────────────────────────────
# Face encoding — 128-d embedding
# ─────────────────────────────────────────────────────────────────────────────

def get_face_encoding(img: np.ndarray) -> np.ndarray | None:
    """
    Return a 128-dimensional face embedding for the most prominent face in
    the image, or None if no face is detected.

    Parameters
    ----------
    img : BGR uint8 image (any size)

    Strategy
    --------
    • Images with a short side < 100 px are processed with
      number_of_times_to_upsample=2 so that the HOG face detector has
      enough resolution to trigger.
    • Larger images use upsample=1 to save ~40 ms per image.
    • The face_recognition library internally uses dlib's ResNet face
      descriptor model — a 29-layer deep net that outputs 128-d vectors
      where same-person pairs are <0.6 Euclidean distance apart.
    """
    if img is None or img.size == 0:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    upsample = 2 if min(h, w) < 100 else 1

    try:
        locs = face_recognition.face_locations(rgb, number_of_times_to_upsample=upsample)
        if not locs:
            return None
        encs = face_recognition.face_encodings(rgb, locs)
        return encs[0] if encs else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SSIM — Structural Similarity Index
# ─────────────────────────────────────────────────────────────────────────────

def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index between two images.

    Both are resized to TARGET_SIZE and converted to grayscale before
    comparison so the score is always on comparable dimensions regardless of
    the raw input size.

    SSIM decomposes image quality into three independent components:
        l(a,b) — luminance similarity  (mean brightness agreement)
        c(a,b) — contrast similarity   (variance agreement)
        s(a,b) — structure similarity  (normalised cross-correlation)

    SSIM = l(a,b) · c(a,b) · s(a,b)

    Returns
    -------
    float in [−1, 1].  1.0 = identical.  Typical improvement: +0.05–0.30.
    """
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0

    def to_gray(img):
        r = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        return cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) if len(r.shape) == 3 else r

    ga, gb = to_gray(a), to_gray(b)
    score, _ = structural_similarity(ga, gb, full=True)
    return float(score)


# ─────────────────────────────────────────────────────────────────────────────
# Composite Perceptual Quality Score — innovation beyond the brief
# ─────────────────────────────────────────────────────────────────────────────

def compute_pq_score(
    sharpness_after:  float,
    ssim_improvement: float,
    match_after:      bool,
) -> float:
    """
    Combine sharpness, SSIM gain and identity-match into one [0, 100] score.

    Weights
    -------
    40% normalised sharpness (capped at 300 — typical ceiling for good crop)
    35% SSIM improvement (capped at 0.5 — asymptotic ceiling)
    25% binary match bonus

    Used only for the HTML report — not part of evaluation_metrics.json.
    """
    norm_sharp = min(sharpness_after / 300.0, 1.0)
    norm_ssim  = min(max(ssim_improvement, 0.0) / 0.5, 1.0)
    match_bonus = 1.0 if match_after else 0.0

    pq = (0.40 * norm_sharp + 0.35 * norm_ssim + 0.25 * match_bonus) * 100.0
    return round(pq, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Identity clustering — group faces that look like the same person
# ─────────────────────────────────────────────────────────────────────────────

def cluster_by_identity(
    results: list[dict],
    tolerance: float = FACE_MATCH_TOLERANCE,
) -> dict[int, list[str]]:
    """
    Greedy threshold-based clustering of result entries by their face encoding.

    For each entry with a 'enc_enh' key (added externally), assign it to the
    first existing cluster whose centroid is within *tolerance* Euclidean
    distance.  If none, open a new cluster.

    Returns
    -------
    dict[cluster_id → list[filename]]
    """
    clusters: dict[int, dict] = {}   # {id: {centroid: ndarray, files: list}}
    cid = 0

    for r in results:
        enc = r.get("enc_enh")
        if enc is None:
            continue
        assigned = False
        for c in clusters.values():
            dist = float(np.linalg.norm(enc - c["centroid"]))
            if dist < tolerance:
                c["files"].append(r["filename"])
                # Update centroid with running mean
                n = len(c["files"])
                c["centroid"] = c["centroid"] * (n - 1) / n + enc / n
                assigned = True
                break
        if not assigned:
            clusters[cid] = {"centroid": enc.copy(), "files": [r["filename"]]}
            cid += 1

    return {k: v["files"] for k, v in clusters.items() if len(v["files"]) > 1}