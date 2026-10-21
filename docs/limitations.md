# Limitations & future work

What this project does **not** claim, and why. Each item is a deliberate scope decision, stated up front.

## Evaluation scope

- **Not leaderboard numbers.** Off-the-shelf COCO YOLO (not fine-tuned on MOT17), 640 px input on 1080p footage,
  distractor classes not filtered. Scores (HOTA ~0.4) are comparable *to each other*, not to MOTChallenge entries.
- **MOT17 training set only (7 sequences).** The official test ground truth is private; there is no test-set claim.
- **One tuning sequence.** Trackers are tuned on the last 25% of MOT17-02 (yolo11s). Held-out means are reported next
  to the full means; generalisation of a single-sequence config is itself a result, but it is a small sample.
- **Sequences are not independent samples.** Six backbones share each sequence, so "42 wins out of 42" is really
  "7 scenes". Paired deltas are descriptive, not significance tests; with 7 scenes a bootstrap CI would be wide.
- **Mirror, not origin.** MOT17 is downloaded from a community Hugging Face mirror (the official host was unreachable).

## Tracking scope

- **ReID is OSNet x0.25 trained on MSMT17** -- a small, off-the-shelf ReID network, not tuned or fine-tuned on MOT17.
- **BoT-SORT here is "BoT-SORT-style":** it reuses this project's `[cx, cy, a, h]` Kalman state (the original uses
  `[cx, cy, w, h]`), and applies CMC by warping state and covariance, without detection masking in the flow estimator.
- **No learned motion or transformer trackers** (e.g. MOTR, tracking-by-attention), no OC-SORT / StrongSORT / Deep OC-SORT.
- **Offline scoring only:** CMC and ReID features are cached, so tracker timings exclude them; live costs are in `emot track`.

## Edge scope

- **One machine.** All latency/RAM numbers are from one Apple M5 laptop; ratios are hypotheses for other hardware.
  INT8 in particular is known to behave very differently across CPUs.
- **No power measurement** (needs elevated privileges); no thermal-throttling study.
- **Quantisation:** static INT8 with MinMax calibration on 66 frames; no QAT, no per-layer sensitivity search.
- **Browser inference** uses single-threaded WASM unless the host sets cross-origin isolation headers (GitHub Pages does not).

## Agent & analytics

- The **Claude agent has been unit-tested with a scripted client but not exercised against the live API** in CI (no
  key available there); the tool layer it calls is covered by tests and by cross-language parity.
- Analytics are in **pixel space**: speeds are body-heights/second (perspective-invariant), but there is no camera
  calibration, so no metres or km/h.
- **Track IDs are not identities.** A person who leaves and returns may get a new ID; "unique people" is an upper bound.

## Future work

- Fine-tune the detector on MOT17/MOT20 and re-run the ladder; add OC-SORT and a learned-motion tracker.
- Static INT8 on x86 (VNNI) and ARM dot-product devices; a Raspberry Pi / Jetson measurement.
- Homography-based ground-plane analytics; multi-camera association (a different, regulated problem -- see
  [Privacy & ethics](privacy-and-ethics.md)).
- Multi-threaded WASM via a service-worker COOP/COEP shim.
