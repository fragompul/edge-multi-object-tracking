# Trackers

Every tracker here is built from the same primitives (`emot/tracking/`): a constant-velocity **Kalman filter** over
`[cx, cy, aspect, height]`, IoU / Mahalanobis / cosine association costs, **Hungarian** assignment, and a shared
track lifecycle (tentative -> confirmed -> lost -> removed).

## ByteTrack (Zhang et al., ECCV 2022) -- `bytetrack.py`

Motion only. The key idea: a low-confidence box is often a real, partly occluded person, not noise.
1. Predict all tracks. Split detections into high (>= `high_conf_threshold`) and low confidence.
2. **Stage 1:** match every track to the high-confidence boxes by IoU (Hungarian).
3. **Stage 2:** match the *leftover tracks* to the low-confidence boxes by IoU -- recovering occluded people
   without ever letting a low-confidence box start a track.
4. Unmatched high-confidence boxes start tentative tracks; tracks unmatched for `max_age` frames die.

## DeepSORT (Wojke et al., ICIP 2017) -- `deepsort.py`

Adds appearance. Association cost = cosine distance between a detection's embedding and the **gallery** of a
track's recent embeddings (minimum over the gallery), after a hard **Mahalanobis gate** on the Kalman prediction.
Two modes:
- **single-stage** (`cascade=False`): one Hungarian assignment over the gated cost -- the v1 baseline.
- **matching cascade** (`cascade=True`): tracks are matched in order of how recently they were seen
  (`age = 1, 2, ...`), each level against only the still-unclaimed detections, so a track seen last frame always
  beats one that vanished 20 frames ago for a contested detection. Leftovers and tentative tracks get a final IoU pass.

## BoT-SORT (Aharon et al., 2022) -- `botsort.py`, `cmc.py`

ByteTrack's two stages, plus:
- **Camera-motion compensation:** Shi-Tomasi corners -> pyramidal Lucas-Kanade -> RANSAC similarity transform, one
  2x3 warp per frame. Every track's state (centre, velocity, height) and covariance is warped before matching, so a
  panning camera no longer looks like every pedestrian accelerating at once.
- **Appearance fusion (optional):** stage-1 cost `min(IoU distance, cosine / 2)`, gated so appearance only rescues
  pairs that are already spatially plausible (`proximity_threshold`) and only when confident (`appearance_threshold`).
  Track appearance is an exponential moving average.

`use_cmc=False, use_reid=False` is *exactly* ByteTrack -- a unit test asserts identical output -- which is what makes
the ablation clean.

## ReID: OSNet (Zhou et al., ICCV 2019) -- `reid/osnet.py`

Omni-scale residual blocks: four parallel streams of 1-4 stacked light 3x3 convolutions (receptive fields 3-9) fused by
a shared channel gate. Written from the paper with `torchreid`-compatible parameter names; the authors' MSMT17
checkpoint (4,101 identities) loads with `strict=True`. Verified against an independent ONNX export (cosine 1.0000).

## Metrics: HOTA -- `evaluation/hota.py`

`HOTA_a = sqrt(DetA_a * AssA_a)` averaged over IoU thresholds a in {0.05..0.95}. DetA is the Jaccard of detections; AssA
averages, over true positives, the Jaccard of each (ground-truth ID, predicted ID) trajectory pair. Pairing uses a
global-alignment-weighted Hungarian match, as in the paper.
