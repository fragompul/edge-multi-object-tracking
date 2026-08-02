# Methodology

## Problem statement

Multi-object tracking (MOT) asks two questions per video frame: *where*
are the objects of interest (detection), and *which object in this frame
is the same object as in the previous frame* (association/identity).
Detection alone is well-studied and largely solved for common object
classes; the harder, more interesting engineering problem -- and the one
that actually determines whether a tracking system is usable for
analytics (counting, dwell time, trajectory analysis) -- is identity
association under occlusion, crowding, and appearance changes. This
project benchmarks the detection side (YOLO backbone speed/accuracy
trade-offs, edge export/quantization) and the association side
(motion-only ByteTrack vs. motion+appearance DeepSORT) independently,
then evaluates the full pipeline end-to-end with standard MOT metrics.

## Dataset

**MOT17** (Milan et al., "MOT16: A Benchmark for Multi-Object Tracking,"
arXiv:1603.00831 -- MOT17 extends MOT16 with refined ground truth and
three parallel public-detection sets per sequence). A **4-sequence
subset** is used: `MOT17-02`, `MOT17-04`, `MOT17-11`, `MOT17-13`, chosen
for visual diversity within a CPU/time budget:

| Sequence | Camera | Setting | Density |
| --- | --- | --- | --- |
| MOT17-02 | Static | Outdoor daytime street | Dense |
| MOT17-04 | Static | Outdoor daytime street (different angle, longer) | Dense |
| MOT17-11 | Moving (handheld) | Indoor shopping mall | Moderate |
| MOT17-13 | Moving (onboard) | Outdoor night, bus interior/street | Moderate |

Only the `-FRCNN` detector-variant folder is downloaded per sequence
(MOT17 repeats each sequence's images 3x under `-DPM`/`-FRCNN`/`-SDP`,
differing only in the bundled public detections, which this project
doesn't use -- it runs its own YOLO detectors instead). See
`docs/limitations.md` for the full scope discussion (11 excluded
sequences, official-host unavailability, etc.).

Sourced from the public Hugging Face mirror `Lekim89/MOT17` (the official
`motchallenge.net` host was unreachable from this environment --
connection refused, verified not a transient blip) via
`src/emot/data/download_mot17.py`.

## Detection: backbone benchmark

Six pretrained, COCO-trained YOLO checkpoints, filtered to the `person`
class (COCO class 0):

- **YOLOv8**: n / s / m (Ultralytics, prior generation)
- **YOLO11**: n / s / m (Ultralytics, current generation at the time of
  writing)

Used off-the-shelf, not fine-tuned on MOT17: the benchmark's purpose is
comparing backbones' existing detection quality and CPU latency, not
producing a MOT17-specialized detector. Detections are cached once per
`(sequence, backbone)` pair (`src/emot/training/run_detection_cache.py`)
and reused by every downstream tracker, since detection is the expensive
step and doesn't depend on which tracker consumes it.

## Tracking: ByteTrack vs. DeepSORT

Both built from a shared constant-velocity Kalman filter
(`src/emot/tracking/kalman.py`, over `[cx, cy, aspect, height]` state --
the standard SORT-family parameterization) and IoU utility
(`src/emot/tracking/iou.py`):

- **ByteTrack** (Zhang et al., ECCV 2022): two-stage association --
  high-confidence detections matched first via Hungarian assignment on
  IoU, then tracks still unmatched are given a second chance against
  low-confidence detections (never used to spawn new tracks). This
  recovers occluded/blurred detections that plain SORT would discard.
- **DeepSORT** (Wojke et al., ICIP 2017): a hard Mahalanobis gate on the
  Kalman filter's predicted position, then cosine distance between a
  detection's appearance embedding and a track's recent embedding
  gallery among gated candidates. The appearance embedding is a frozen,
  ImageNet-pretrained ResNet18's global-average-pooled features (512-d,
  L2-normalized) rather than a dedicated person-ReID network -- a
  documented simplification, see `docs/limitations.md`.

Both trackers' association thresholds (confidence/IoU/cosine-distance
gates, `min_hits`, `max_age`) are tuned with **Optuna**
(`src/emot/training/tune_trackers.py`), maximizing MOTA on a **temporal**
val split (the trailing 25% of frames in the tuning sequence -- never a
random per-frame split, which would leak the same track's future/past
context across train and val).

## Evaluation

**py-motmetrics**, the standard CLEAR-MOT + IDF1 reference implementation
(`src/emot/evaluation/mot_metrics.py`), reporting MOTA, MOTP, IDF1,
ID-switch count, false positives, misses, and mostly-tracked/mostly-lost
counts. Ground truth is filtered to `class == 1` (pedestrian) with
`consider_flag == 1`, matching MOT17's own evaluation convention.

## Edge export & quantization benchmark

Each backbone is exported to ONNX (static shape, simplified graph) and
dynamically quantized to INT8 via `onnxruntime.quantization`. Latency,
model size, and a detection-agreement proxy (mean best-match IoU against
the PyTorch FP32 reference, not a full mAP -- see
`docs/limitations.md`) are benchmarked across all three variants
(`src/emot/export/benchmark.py`), CPU-only.

## Analytics

Derived from tracked positions, not re-detected: per-track trajectories,
a configurable line-crossing counter (directed, via 2D cross-product
sign), and a 2D occupancy heatmap -- all computed live in the dashboard
from cached detections replayed through ByteTrack (cheap: no neural
network inference in the replay, just Kalman + Hungarian matching), so
this doesn't require its own pre-computed results cache.

## Compute environment

CPU-only laptop (8 cores, ~7.4GB RAM, no GPU). Every stage is sized to
finish in hours: detection caching is the dominant cost (~2-3h for the
full `4 sequences x 6 backbones` grid at CPU-only YOLO speeds), reused
across all downstream tracking/tuning/evaluation runs rather than
re-incurred per experiment.
