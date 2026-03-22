"""
core/config.py — Shared constants for the face enhancement pipeline.

MediaPipe Face Mesh uses 468 3-D landmarks on a normalised [0,1] grid.
The indices below were selected to form tight convex hulls around each
facial region so we can build a smooth blending mask in Stage 4.
"""

from pathlib import Path

# ── Output geometry ──────────────────────────────────────────────────────────
TARGET_SIZE = (240, 240)          # final output WxH (pixels)

# ── Pipeline thresholds ───────────────────────────────────────────────────────
SHARPNESS_GATE       = 80.0       # Laplacian variance: skip enhancement if above
FACE_MATCH_TOLERANCE = 0.60       # face_recognition Euclidean distance threshold

# ── NLM denoising ────────────────────────────────────────────────────────────
NLM_H_DEFAULT        = 8          # luminance denoising strength
NLM_H_COLOR          = 8          # chrominance denoising strength
NLM_TEMPLATE_WINDOW  = 7          # patch size for similarity
NLM_SEARCH_WINDOW    = 21         # search radius

# ── CLAHE ────────────────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT     = 3.5
CLAHE_TILE_GRID      = (4, 4)

# ── Upscaling ────────────────────────────────────────────────────────────────
SMALL_FACE_THRESHOLD = 64         # short-side threshold for 2-pass upscale

# ── Unsharp mask presets ──────────────────────────────────────────────────────
UNSHARP_INTERMEDIATE = (1.0, 1.6)   # (sigma, strength) between the two 2× passes
UNSHARP_ZONE_HOT     = (0.8, 2.0)   # eye + nose region
UNSHARP_ZONE_COOL    = (1.2, 1.3)   # rest of face
UNSHARP_FALLBACK     = (1.0, 1.5)   # whole face when no landmarks

# ── MediaPipe face-mesh detection ─────────────────────────────────────────────
MP_MIN_DETECTION_CONF = 0.3
MP_REFINE_LANDMARKS   = True

# ── Mask feathering ───────────────────────────────────────────────────────────
MASK_FEATHER_KERNEL  = 21         # GaussianBlur ksize for soft zone boundary

# ── Face alignment ────────────────────────────────────────────────────────────
ALIGN_MAX_ANGLE_DEG  = 35.0       # ignore rotations > this (profile/extreme pose)

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Face Mesh landmark indices — selected for region coverage.
# Source: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# ─────────────────────────────────────────────────────────────────────────────

# Left eye outer contour (anatomically left = camera right)
LEFT_EYE = [
    362, 382, 381, 380, 374, 373, 390,
    249, 263, 466, 388, 387, 386, 385, 384, 398,
]

# Right eye outer contour
RIGHT_EYE = [
    33,   7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]

# Left eye iris (MediaPipe refine_landmarks=True adds irises at 468–477)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Nose bridge, tip and alae
NOSE_BRIDGE = [6, 197, 195, 5]
NOSE_TIP    = [1, 2, 98, 327, 326, 2]
NOSE_ALAE   = [129, 203, 206, 216, 212, 358, 423, 426, 432, 399]
NOSE_ALL    = list(set(NOSE_BRIDGE + NOSE_TIP + NOSE_ALAE))

# Eyebrows (for extended "HOT" zone)
LEFT_EYEBROW  = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW = [46,  53,  52,  65,  55,  70,  63, 105,  66, 107]

# Combined "hot" zone: eyes + eyebrows + nose
HOT_ZONE = list(set(LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + NOSE_ALL))

# Eye-centre indices used for alignment
LEFT_EYE_CENTRE_IDX  = 468   # iris centre (requires refine_landmarks)
RIGHT_EYE_CENTRE_IDX = 473
# Fallback if iris not detected (refine_landmarks=False)
LEFT_EYE_FALLBACK    = [362, 263]
RIGHT_EYE_FALLBACK   = [33, 133]