# Web app: EMOT Studio

**Live:** <https://fragompul.github.io/edge-multi-object-tracking/> -- a static site (GitHub Pages), no backend.

| Page | What it does |
| --- | --- |
| Overview | Hero clip with live overlay; headline findings computed in the browser from `results.json` |
| Tracker Arena | Seven MOT17 clips x five trackers; side-by-side compare from **one decoded video** (panels draw the same frames, so they cannot drift); identity-switch markers and timeline ticks |
| Benchmarks | Sortable leaderboard, tracker x scene heatmap, CMC/ReID ablation, backbone Pareto -- filter by camera type and backbone |
| Edge Lab | Latency / RAM / accuracy per runtime, thread scaling, resolution Pareto, the methodology correction |
| Analytics Studio | Draw a line or zone on tracked video; crossings, dwell, loitering, occupancy timeline, heatmap; "Ask EMOT" offline planner |
| Live in browser | YOLO11n (FP32 or INT8) + ByteTrack on webcam / sample / your video via ONNX Runtime Web (WebGPU or WASM) |

## Stack

React 18 + TypeScript + Vite, hand-written SVG charts (d3-scale/shape only for maths), CSS design tokens mirrored from
`emot.viz.palette`. `onnxruntime-web` is loaded on demand (WebGPU build), the WASM runtime is served from `/ort/`.

## Cross-language parity

The browser does real work, so it is tested against Python:

```
scripts/make_ts_fixtures.py   ->  web/src/core/__fixtures__/{bytetrack,analytics,yolo}.json
web/src/core/parity.test.ts   ->  18 tests: ByteTrack on 150 real frames, every analytics tool,
                                  the offline planner, YOLO decode/NMS/letterbox geometry, Hungarian optimality
```

Writing these tests caught a real bug in the Python NMS (see [Experiments](experiments.md), Phase 11).

## Data pipeline

`scripts/export_web_data.py` writes `web/public/data/` (committed, ~9 MB): 10-second H.264 clips (the window with the
most ByteTrack identity switches in each sequence -- the *hard* part, stated in the UI), per-tracker tracks, raw
detections, ground truth, identity-switch events, and `results.json`. The site builds in CI without MOT17 or weights.

## Run / build / deploy

```bash
cd web && npm install && npm run dev      # http://localhost:5173
npm test                                  # parity tests
npm run build                             # -> web/dist
```

Deployment is `.github/workflows/pages.yml`: tests + build the app, build these docs into `/docs/`, publish to Pages.
One-time: repository **Settings -> Pages -> Source: GitHub Actions**.

## Measured in-browser performance

On an Apple M5 laptop in Chrome: ~10 FPS on single-threaded WebAssembly (about 90 ms of inference per frame) and ~55-75 FPS on
WebGPU (10-15 ms). Static hosting cannot set the COOP/COEP headers WASM threading needs. Your hardware will differ.
