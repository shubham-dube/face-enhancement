# Architecture & Implementation Notes
## Sentio Mind · Project 4 · CCTV Face Enhancement

---

## What This Project Actually Does

CCTV cameras produce face crops as small as 12×12 pixels. At that size, face recognition fails, images look like blurry blobs, and any profile photo is unusable. This pipeline takes those tiny degraded crops and produces clean 240×240 outputs that a face recognition model can reliably match.

The constraint that makes this genuinely difficult: **no deep learning**. No ESRGAN, no GFPGAN, no CodeFormer. Everything is classical computer vision — and that forces you to understand the problem deeply rather than throwing a model at it.

---

## The Actual Problem

A face from CCTV footage has passed through a chain of lossy transforms before you ever see it:

```
Real face
  → cheap lens          (diffraction blur, chromatic aberration)
  → small sensor        (shot noise, read noise)
  → H.264 compression   (DCT blocking artifacts at 8×8 boundaries)
  → face detector crop  (misalignment, partial face, background bleed)
  → frame extraction    (motion blur if subject moved)
  = raw_faces/*.jpg     (12–80 px, noisy, blurry, compressed)
```

The pipeline reverses these in the correct order. Doing them out of order makes results worse — denoising after sharpening amplifies noise, CLAHE before denoising over-enhances noise patterns, etc.

---

## Pipeline Overview

```
Input image
    │
    ├─ sharpness > 80 AND size > 240px? → resize + recovery sharpen → output
    ├─ sharpness > 80?                  → resize only              → output
    ├─ size > 64px AND sharpness > 20?  → CLAHE + resize + zone sharpen → output
    └─ tiny or heavily blurred          → full 4-stage pipeline    → output
```

The routing matters as much as the stages themselves. Applying NLM denoising to a 150px crop that doesn't need it will blur it and reduce sharpness — the opposite of the goal.

---

## Stage 1 — Denoising

**For tiny faces (≤ 32 px):** Bilateral filter instead of NLM.

NLM (Non-Local Means) works by searching the image for similar patches and averaging them. On a 12×12 face there are no meaningful non-local patches — every patch overlaps everything else. The algorithm averages edges away. Bilateral filter uses spatial proximity and pixel similarity instead, which preserves the few edges that exist at this scale.

**For larger faces:** Adaptive NLM where the `h` parameter scales with estimated noise level.

Noise is estimated using the Median Absolute Deviation of the Laplacian response (Donoho-Johnstone 1994). A heavily noisy image gets h≈10; a cleaner one gets h≈4. The fixed h=8 from the brief is the midpoint of this range.

**For medium-quality images (sharpness 20–80, size > 64px):** Stage 1 is skipped entirely. These images don't need denoising — they need sharpening. Running NLM on them costs 80ms and makes them slightly blurrier.

---

## Stage 2 — CLAHE

Contrast Limited Adaptive Histogram Equalization on the L channel of LAB colour space.

LAB is used because it separates luminance (L) from colour information (a, b). Boosting contrast in L does not change hue or saturation — skin tones stay natural. Doing the same operation in BGR or HSV distorts colour.

The tile grid adapts to image size: `tile = max(1, short_side // 8)`. A fixed (4,4) grid on a 12px image creates 3×3 pixel tiles — a local histogram over 9 pixels is meaningless. With an adaptive grid, tiles always contain enough pixels for a statistically valid histogram.

clipLimit=3.5 was chosen empirically: 2.0 is too conservative for very flat CCTV images, 5.0 over-enhances and makes skin look patchy.

---

## Stage 3 — Upscaling

**For tiny faces (≤ 32 px):** Two IBP passes with intermediate unsharp sharpening.

Iterative Back-Projection (Irani & Peleg, 1991) is the key upgrade over a simple LANCZOS resize. The algorithm:

```
1. Start: LANCZOS4 upscale to target size
2. For N iterations:
   a. Simulate degradation: downscale current estimate back to original size
   b. Measure residual: residual = original - simulated
   c. Project residual up: residual_up = CUBIC resize of residual
   d. Update: current = current + 0.5 × residual_up
```

Each iteration forces the high-res estimate to be consistent with the original observation. After 8 iterations the result has significantly better edge sharpness than a single-step interpolation. The gain of 0.5 is conservative — higher values cause oscillation.

The two-pass structure (3× IBP → unsharp → IBP to larger size → LANCZOS to 240) never asks for more than ~4× per step, staying within the reliable range of each method.

**For small faces (32–64 px):** One IBP pass then LANCZOS to target.

**For larger faces:** Direct LANCZOS4 resize. These images have enough signal that IBP's benefit is marginal relative to its compute cost.

---

## Stage 4 — Zone Sharpening

MediaPipe Face Mesh detects 468 3D landmarks on the face. We use these to identify the eye and nose region, build a convex hull mask, feather the edges with a Gaussian blur, then apply different sharpening strengths to the two zones.

**Why different strengths per zone:**

The eye and nose region drives face recognition. These areas contain the features (iris texture, eye corner shape, nostril shadow, nose bridge edge) that dlib's 128-d encoder actually uses. Sharpening them more aggressively extracts more useful signal.

The rest of the face (forehead, cheeks, chin) contributes less to recognition and is more susceptible to looking artificial when over-sharpened — pore noise gets amplified.

**Adaptive strength:** The sharpening strength is not fixed. It scales inversely with the current Laplacian variance of the image:

```python
hot_strength = np.interp(current_lv, [10.0, 150.0], [1.4, 0.4])
```

A blurry image at sharpness 30 gets hot_strength=1.3. An already-decent image at sharpness 120 gets 0.5. This prevents the stacked-operations problem where fixed-strength sharpening on an already-sharp image creates ringing artifacts with Laplacian variance > 1000.

**Auto-alignment:** Before building the zone mask, the face is rotated so the eye axis is horizontal. dlib's landmark predictor (used inside face_recognition) was trained on aligned faces. A 15° tilt meaningfully degrades the 128-d encoding. The rotation uses iris landmark indices 468 and 473 (available with `refine_landmarks=True`) and is skipped if the angle exceeds 35° (likely a profile view where forced alignment would make things worse).

**Laplacian pyramid blending:** The hot and cool sharpened images are merged using a 3-level Laplacian pyramid rather than simple alpha blending. Alpha blending produces a visible halo at the zone boundary. Pyramid blending composites each spatial frequency band separately, so the boundary is perceptually invisible.

**Sharpness ceiling:** After all operations, if the Laplacian variance exceeds 450 the image has ringing artifacts, not real detail. A proportional Gaussian smooth brings it back to a natural range without re-blurring real edges.

---

## Face Recognition Matching

Three strategies applied in sequence before giving up on a match:

**1. dlib at adaptive tolerances.** Standard `compare_faces` is tried at [0.48, 0.54, 0.60]. Starting strict eliminates false positives; relaxing to 0.60 catches faces just outside the default threshold.

**2. DeepFace / Facenet as fallback.** When dlib gets no match, a Facenet embedding is computed via DeepFace and compared against pre-computed reference embeddings using cosine similarity. Facenet is significantly more robust to low-quality inputs than dlib because it was trained on a much larger and more varied dataset. Threshold: cosine similarity > 0.50.

**3. Reference encodings built once at startup.** Both dlib and DeepFace reference encodings are computed once before the main loop. Computing them inside the per-face loop would be O(N×M) encoder calls for N faces and M references.

---

## File Layout

```
solution.py          main script — all template stubs, correct signatures
extract_faces.py     standalone video → raw_faces/ tool

core/
  config.py          constants and MediaPipe landmark indices
  stages.py          pipeline stages 1–4 with all logic
  metrics.py         sharpness, face encoding, SSIM, matching, clustering
  report.py          self-contained HTML report generator
```

`solution.py` imports from `core/`. The function signatures in `solution.py` are exactly as specified in the template — no renames, no additions to the public API.

---

## Output Schema

`evaluation_metrics.json` matches the provided schema exactly. No keys are added, removed, or renamed. The `enc_enh` field used internally for HTML report clustering is stripped from the `per_face` array before serialisation and never appears in the JSON.

---

## Performance Notes

| Operation | Per-face | ×100 faces |
|---|---|---|
| imread | ~2ms | 0.2s |
| Bilateral / NLM (adaptive) | 5–80ms | 0.5–8s |
| CLAHE | ~5ms | 0.5s |
| IBP 8 iterations (tiny path) | ~55ms | 5.5s |
| LANCZOS resize | ~3ms | 0.3s |
| MediaPipe mesh (singleton) | ~60ms | 6s |
| face_recognition encode | ~80ms | 8s |
| SSIM | ~10ms | 1s |
| **Total** | | **~22–24s** |

The MediaPipe singleton is the most impactful optimization — loading the TFLite model per image would add 60ms × 100 = 6 seconds. The model is initialized once and reused across all faces. `cleanup_mesh()` is called at the end of main to release the resources cleanly.

The medium-quality path (skip Stage 1) saves ~80ms per image. For a batch where most faces are in the 40–80 sharpness range, this can shave 5–6 seconds off the total runtime.

---