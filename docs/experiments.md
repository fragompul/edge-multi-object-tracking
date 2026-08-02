# Experiments

This log describes each phase in the order it actually happened,
including the real bugs found along the way -- a real project's
experiment log has dead ends and fixes in it, and hiding them would make
this read like a tutorial instead of a research report.

## Phase 0 -- dataset acquisition

`motchallenge.net` (the official MOT17 host) was unreachable from this
development environment (`curl` returned "no route to host", not a
timeout -- checked, not a transient blip). Switched to the public
Hugging Face mirror `Lekim89/MOT17`, verified it served real JPEG content
(not LFS pointer stubs) behind its resolve URLs before committing to it.

**Bug 1 -- HF CDN rate-limiting under concurrency.** The first download
attempt used 12 concurrent worker threads with immediate 3-retry-no-delay
on failure. This worked for the first ~2900/3308 files, then hit
persistent HTTP 429 ("Too Many Requests") -- and because retries had no
backoff, each retry immediately re-triggered the same 429, so the job
finished with 5 permanently-failed files instead of recovering. Fixed by
adding exponential backoff with jitter (`min(30s, 2^attempt * 0.5s) +
random(0, 0.5s)`) and reducing worker count to 3-5; a rerun (idempotent,
skips already-downloaded files) picked up the 5 stragglers with zero
failures.

## Phase 1 -- core algorithm implementation + smoke tests

ByteTrack and DeepSORT were both built from shared primitives (Kalman
filter, IoU matrix, Track lifecycle) rather than wrapped from an existing
tracking library, then smoke-tested against a synthetic single-object
moving-box sequence before any real data was involved -- both correctly
assigned one persistent track ID across all frames and confirmed at the
expected `min_hits` frame.

**Bug 2 -- `py-motmetrics` crashes outright under NumPy 2.x.** Wiring up
the first real accuracy check (`MOTAccumulator.update`) immediately threw
`AttributeError: np.asfarray was removed in the NumPy 2.0 release` from
inside `motmetrics.distances.iou_matrix` -- not a subtle numerical
divergence, a hard crash on the very first call, because this project's
`numpy>=1.26` (no upper bound) dependency resolved to NumPy 2.x while
`py-motmetrics` 1.4.0 still calls a function removed two major versions
ago. Fixed by reimplementing the ~10-line IoU-to-distance conversion
locally, reusing this project's own already-tested `emot.tracking.iou`
utility instead of depending on an upstream fix landing. Verified against
a synthetic test with a known expected ID-switch count before trusting
it on real data.

## Phase 2 -- test-driven bug: empty-trajectory heatmap crash

While writing `tests/unit/test_analytics.py`, a deliberately-included
edge case (`occupancy_heatmap({})`  -- no tracks at all) crashed with
`ValueError: need at least one array to concatenate`. The function had a
dead `if len(all_points) == 0: return zeros(...)` guard placed *after*
the `np.concatenate` call that was supposed to protect it -- the
early-return check needed to happen before concatenation, not after.
Fixed by checking for an empty input list before concatenating rather
than checking the (never-reachable) result afterward. A concrete example
of why the edge case belongs in the test suite before the dashboard ships
a page that calls this function on a sequence with a still-empty
detection cache.

## Phase 3 -- proactive resumability fix (not a bug that fired, one that would have)

`run_detection_cache.py`'s summary metadata (per-backbone latency/count
stats) was originally written to `summary.json` only once, after the
*entire* multi-hour loop finished. Caught during review, before it could
actually bite: an interruption near the end of the ~2-3 hour detection
caching run would have lost all summary metadata even though the
per-file `.npz` detection caches (the actually expensive artifacts) were
safely resumable. Fixed to write `summary.json` incrementally after every
`(sequence, backbone)` pair completes, matching the resumability
guarantee the rest of the pipeline already had.

## Phase 4 -- `.gitignore` bug: results were being silently excluded

Copying the `.gitignore` pattern from this portfolio's
`industrial-visual-anomaly-detection` project (which does correctly
ignore `/results/`... except that project's `results/` holds only
model-training metrics, not this project's much larger detection cache)
included `/results/` verbatim, without adjusting it for this project's
different directory layout, where `results/` holds the small JSON
metrics that should be committed and `cache/` holds the large
regenerable detection arrays that shouldn't. Caught immediately at the
first `git add` after hyperparameter tuning finished (`git status`
reported the JSON file as "ignored," not just "untracked" -- a signal
worth noticing rather than force-adding past). Fixed by ignoring
`/cache/` instead of `/results/`.

## Phase 5 -- full detection cache + hyperparameter tuning

Ran `run_detection_cache.py` for the full `4 sequences x 6 backbones`
grid (~3300 frames x 6 backbones ≈ 2.5 hours wall-clock on 8 CPU cores,
the dominant cost in this project) and `tune_trackers.py` (30 Optuna
trials each for ByteTrack and DeepSORT, on `MOT17-02`/`yolo11s`'s val
split) concurrently -- CPU-contended but both are independently bounded,
hours-scale jobs, and neither depends on the other's output. Tuned MOTA
on the val split landed at ~0.30 for both trackers -- a real, unglamorous
number from off-the-shelf detectors on real crowded MOT17 footage, not
adjusted to look better.

## Phase 6 -- full benchmark grid + export benchmark

`run_tracking_benchmark.py` (48 runs, replaying cached detections through
both tuned trackers) and `run_export_benchmark.py` (6 backbones x 3
variants: PyTorch FP32, ONNX FP32, ONNX INT8) both completed without
further implementation bugs -- but produced the two most substantively
interesting *results* of the project (not bugs, genuine findings):
ByteTrack/DeepSORT's MOTA-vs-IDF1 disagreement on MOT17-04, and ONNX/INT8
export **not** improving CPU latency despite reliably shrinking file
size. Both are discussed in full in `docs/results.md` rather than here,
since they're findings about the domain, not about this codebase.
