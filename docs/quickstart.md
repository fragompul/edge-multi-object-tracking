# Quickstart

```bash
git clone https://github.com/fragompul/edge-multi-object-tracking && cd edge-multi-object-tracking
python -m venv .venv && source .venv/bin/activate        # Python 3.11+
pip install -e ".[dev,dashboard,agent]"
```

## Track your own video (no dataset needed)

```bash
emot track my_video.mp4 --tracker botsort --save out.mp4 --jsonl tracks.jsonl
emot track 0 --show                                      # webcam, live preview (q to quit)
emot track rtsp://camera/stream --tracker bytetrack      # network stream
emot track data/mot17/MOT17-04 --max-frames 200          # a MOT17 image folder
```

YOLO weights are downloaded to `weights/` on first use. Trackers: `bytetrack`, `botsort`, `botsort_cmc`, `deepsort`,
`deepsort_osnet_cascade`. Detectors: any Ultralytics checkpoint (`--model yolo11s.pt --imgsz 960`) or an exported
ONNX model (`--onnx model.onnx --provider coreml --threads 4`).

## Ask questions about the result

```bash
emot ask tracks.jsonl "How many people crossed the middle line?" --fps 30 --width 1920 --height 1080 --offline
```

## Reproduce the benchmark

```bash
make data        # 7 MOT17 sequences (~2 GB of frames)
make detect      # YOLO on every (sequence, backbone)     ~10 min on Apple GPU
make features    # ReID embeddings + camera motion        ~15 min
make tune        # Optuna, HOTA objective                 seconds
make bench       # 294 tracking runs + detection AP       ~5 min
make edge        # edge benchmark (idle machine!)         ~12 min
make figures web-data
```

## Explore

```bash
make dashboard                      # Plotly Dash lab on http://127.0.0.1:8050
cd web && npm install && npm run dev   # the Studio web app
docker compose up dashboard         # containerised dashboard
```
