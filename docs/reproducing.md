# Reproducing every claim

| Claim | How to regenerate / verify |
| --- | --- |
| Leaderboard, ablations, per-sequence tables | `make bench` then `python scripts/make_results_docs.py` |
| HOTA is correct | `pytest tests/unit/test_hota.py`; TrackEval cross-check procedure in [Experiments](experiments.md), Phase 7 |
| OSNet is correct | strict state-dict load; cosine 1.0 vs an independent ONNX export; `pytest tests/unit/test_reid.py` |
| ONNX detector == PyTorch | `pytest tests/integration` (real YOLO11n, IoU > 0.97 on the Ultralytics bus image; > 0.999 on MOT17) |
| Browser tracker/analytics == Python | `python scripts/make_ts_fixtures.py && (cd web && npm test)` |
| Edge numbers | `make edge`, `python scripts/profile_onnx_ops.py` (idle machine; results are machine-specific) |
| Camera-motion labels | `python scripts/build_sequence_table.py` |
| Figures / GIFs / banner | `make figures`, `scripts/make_gifs.py`, `scripts/make_brand_assets.py` |

Everything under `results/` is committed so that docs and the web app always match the data in the repository.
