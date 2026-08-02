# Limitations & Future Work

Honest accounting of what this project does *not* claim, and why -- every
constraint below is a deliberate scope decision made explicit, not a gap
discovered after the fact and hidden.

## Dataset scope

- **4 of MOT17's 7 train sequences** (`MOT17-02`, `-04`, `-11`, `-13`),
  not the full benchmark, chosen for visual diversity (static/moving
  camera, indoor/outdoor, day/night, dense/moderate crowding) within a
  CPU/RAM/time budget. Results here are this project's actual
  measurements on this subset, not a claim about "what YOLO/ByteTrack/
  DeepSORT score on MOT17" in the official-leaderboard sense.
- **Sourced from a community Hugging Face mirror** (`Lekim89/MOT17`), not
  `motchallenge.net` directly -- the official host was unreachable from
  this development environment (connection refused, verified not a
  transient failure, not investigated further since a working public
  mirror with the canonical directory structure was available).
- **Test-set ground truth is not used** (MOT17's official test split
  keeps GT private for challenge-submission integrity, standard MOT
  protocol) -- all evaluation is on the train split's own ground truth,
  with a temporal train/val split reserved for hyperparameter tuning.

## Detection scope

- **Off-the-shelf COCO-pretrained YOLO, not fine-tuned on MOT17.** The
  benchmark's purpose is comparing backbone speed/accuracy trade-offs and
  edge-export behavior, not producing a MOT17-specialized detector. A
  fine-tuned detector would likely score higher on both detection and
  downstream tracking metrics; that's a different project (detector
  training), not this one (backbone/tracker/export benchmarking).
- **No detection-only mAP.** The "accuracy" axis of the backbone
  benchmark uses downstream tracking MOTA (ByteTrack, held fixed) as the
  accuracy signal rather than a separate box-level mAP against MOT17
  ground truth, since that would need its own IoU-matching/mAP machinery
  in addition to what tracking evaluation already provides. Defensible
  since tracking quality is the metric that matters end-to-end, but it
  does mean a backbone that's slightly better at raw detection but worse
  at whatever ByteTrack does with its output could look worse here than
  it "really" is at detection alone.

## Tracking scope

- **DeepSORT's appearance embedding is a frozen, generic
  ImageNet-pretrained ResNet18**, not a dedicated person re-identification
  network trained with a metric-learning loss (triplet/contrastive) on a
  ReID dataset like Market-1501. This is the approach used before
  dedicated ReID nets became standard in the DeepSORT lineage, and avoids
  an extra pretrained-weights dependency for a CPU-only project -- but it
  is a real gap: a proper ReID embedding is specifically optimized so
  "same identity" pairs are close and "different identity" pairs are far
  apart, which a generic classifier's features only approximate. The gap
  this leaves is measured directly (DeepSORT's ID-switch count vs.
  ByteTrack's, in `docs/results.md`), not assumed away.
- **DeepSORT association uses a single Hungarian assignment stage**, not
  the original paper's cascaded per-track-age matching (which prioritizes
  recently-seen tracks when resolving simultaneous ambiguous matches in
  dense crowds). A real simplification with a real effect in the
  heaviest-crowding sequences, not a hidden shortcut.
- **Hyperparameter tuning is done on one representative sequence**
  (`MOT17-02` by default), not all 4, to bound Optuna trial compute
  (DeepSORT trials in particular re-run appearance embedding per trial).
  The tuned config is then evaluated on the other 3 sequences too, so how
  well a single-sequence-tuned config generalizes is itself an honestly
  reported result, not an assumption.

## Export/quantization scope

- **Dynamic, not static/calibrated, INT8 quantization.** Static
  quantization needs a representative calibration dataset run through the
  model to collect activation ranges -- disproportionate setup for a
  benchmark whose point is "how much latency/size do you get for free,"
  not squeezing out the last percent of accuracy. The accuracy cost of
  this choice is measured directly (`mean_iou_vs_reference` in the export
  benchmark), not assumed to be zero.
- **Detection-agreement proxy, not mAP**, for measuring how much
  quantization changes what the model detects (mean best-match IoU
  between the FP32 reference and each variant, per frame, averaged) --
  same reasoning and same trade-off as the backbone-benchmark accuracy
  metric above.

## What a production system would add

- Fine-tuned detector on real MOT17 (or a similar target-domain) training
  data, rather than off-the-shelf COCO weights.
- A dedicated person-ReID embedding network for DeepSORT (e.g. an
  OSNet/BoT variant trained on Market-1501 or a similar ReID dataset).
- Full cascaded DeepSORT matching (track-age-prioritized), not the
  single-stage simplification used here.
- Static, calibrated INT8 quantization (or QAT) if the measured dynamic-
  quantization accuracy cost proves too high for the target deployment.
- GPU-accelerated inference for real-time (>15fps) throughput on the
  larger backbones -- this project's CPU-only latency numbers characterize
  the actual edge-deployment target, but a GPU would change which
  backbone size is "fast enough."
- Multi-camera / re-identification across camera views (this project is
  single-camera, single-video tracking only).
