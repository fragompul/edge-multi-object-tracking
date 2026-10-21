# Changelog

All notable changes. Format: [Keep a Changelog](https://keepachangelog.com/), versions follow [SemVer](https://semver.org/).

## [1.0.0] - 2026-10-21

The portfolio rebuild: a re-run of everything with more data, a real ReID network, camera-motion compensation, a
corrected edge benchmark, and three new front-ends.

### Added
- **BoT-SORT** (`emot.tracking.botsort`) with **camera-motion compensation** (`emot.tracking.cmc`) and optional ReID fusion.
- **OSNet** ReID network written from the paper (`emot.reid`), loading the official MSMT17 weights with `strict=True`.
- **DeepSORT matching cascade** (`cascade=True`), vectorised gating (identical output to the previous implementation).
- **HOTA / DetA / AssA / LocA** (`emot.evaluation.hota`), validated against TrackEval to 10 decimals; **detection AP**.
- **All 7 MOT17 training sequences**, 6 backbones, 7 tracker variants (294 runs), plus an **input-resolution sweep**.
- **`emot` CLI** (`track`, `ask`, `results`) with a threaded real-time pipeline (files, webcam, RTSP, image folders).
- **ONNX Runtime detector** with explicit provider / threads and Python-side letterbox, decode, NMS.
- **Edge benchmark v2**: one process per configuration, AP measured per row, CoreML, static INT8, threads, resolution, operator profile.
- **Scene analytics** (zones, dwell, loitering, crowding, speed in body-heights/s) and a **natural-language agent**
  (Claude tool use + offline planner).
- **EMOT Studio** web app (React/TypeScript): Tracker Arena, Benchmarks, Edge Lab, Analytics Studio, in-browser YOLO + tracking
  (WebGPU/WASM), with cross-language parity tests.
- Docs site (mkdocs-material), model & dataset cards, privacy & ethics, brand assets, Dockerfile, Makefile, pre-commit,
  Pages / release workflows.

### Changed
- Tuning objective MOTA -> **HOTA**; detections cached to a 0.1 confidence floor (was 0.25); results regenerated from scratch.
- Dashboard re-themed and updated to the new result schema.

### Fixed
- **Edge benchmark compared different input shapes** (PyTorch 384x640 vs ONNX 640x640): replaced.
- **Static INT8 scored AP 0.0** (decode graph quantised): decode kept in float.
- **NMS `top_k` truncated candidates before suppression** (found by the Python/TypeScript parity tests).
- Author name inconsistency in package metadata.

### Removed
- `run_export_benchmark.py` and `export/benchmark.py` (superseded by the edge benchmark v2).

## [0.1.0]
Initial release: ByteTrack vs DeepSORT on 4 MOT17 sequences, YOLOv8/11 backbones, ONNX/INT8 export, Dash dashboard.
