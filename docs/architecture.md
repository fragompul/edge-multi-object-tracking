# Architecture

## Component overview

```mermaid
flowchart TB
    subgraph Data
        MOT17["MOT17 subset<br/>(4 sequences, HF mirror)"]
    end

    subgraph Detection
        YOLO["YOLOv8 / YOLO11<br/>(n/s/m, COCO-pretrained)"]
        Cache["Detection cache<br/>(cache/detections/*.npz)"]
    end

    subgraph Tracking
        BT["ByteTrack<br/>(Kalman + two-stage IoU)"]
        DS["DeepSORT<br/>(Kalman + Mahalanobis gate<br/>+ ResNet18 appearance)"]
        Tune["Optuna tuning<br/>(val-split MOTA)"]
    end

    subgraph Evaluation
        MM["py-motmetrics wrapper<br/>(MOTA/IDF1/switches)"]
        Results["results/*.json"]
    end

    subgraph Export
        ONNX["ONNX export"]
        Quant["INT8 dynamic quantization"]
        Bench["Latency/size/quality benchmark"]
    end

    subgraph Analytics
        Traj["Trajectories"]
        Count["Line-crossing counts"]
        Heat["Occupancy heatmap"]
    end

    subgraph Dashboard
        Dash["Plotly Dash<br/>(5 pages)"]
    end

    MOT17 --> YOLO --> Cache
    Cache --> BT & DS
    Tune -.tuned config.-> BT & DS
    BT & DS --> MM --> Results
    YOLO --> ONNX --> Quant --> Bench
    Cache --> Traj --> Count & Heat
    Results --> Dash
    Bench --> Dash
    Traj & Count & Heat --> Dash
    Cache --> Dash
```

## Why detections are cached separately from tracking

Detection (a full YOLO forward pass per frame) is the expensive step;
tracking (Kalman prediction + Hungarian assignment) is cheap by
comparison. The benchmark grid is `sequences x backbones x trackers`, but
which tracker runs downstream has no effect on what the detector
produces -- so `run_detection_cache.py` runs `sequences x backbones`
detection passes once, and `run_tracking_benchmark.py` replays the cached
detections through both trackers, roughly halving the total detection
compute versus re-running YOLO once per tracker. The dashboard's
Sequence Explorer and Analytics pages also read directly from this cache,
so exploring the data doesn't require the detector to be loaded at all.

## Inference/evaluation sequence (one benchmark cell)

```mermaid
sequenceDiagram
    participant Runner as run_tracking_benchmark.py
    participant Cache as Detection cache (.npz)
    participant Tracker as ByteTrack / DeepSORT
    participant ReID as AppearanceEmbedder (DeepSORT only)
    participant Acc as MOTAccumulator
    participant GT as gt.txt

    Runner->>Cache: load cached detections for (sequence, backbone)
    loop each frame
        Runner->>Tracker: update(detections[, embeddings])
        opt DeepSORT
            Tracker->>ReID: embed_crops(detection crops)
            ReID-->>Tracker: 512-d embeddings
        end
        Tracker-->>Runner: confirmed tracks (xyxy, track_id)
        Runner->>GT: ground truth for this frame
        Runner->>Acc: update(gt_boxes, predicted_tracks)
    end
    Runner->>Acc: compute() -> MOTA/IDF1/switches/...
    Acc-->>Runner: SequenceMetrics
    Runner->>Runner: write results/<seq>/<backbone>/<tracker>.json
```

## Repository layout

Deliberately different from this portfolio's
`industrial-visual-anomaly-detection` (which splits into a FastAPI
backend + a separate React frontend service): this project is a single
Python codebase, since a Dash app *is* Python and doesn't need a separate
API service to talk to a separate frontend build.

```
src/emot/
  data/         MOT17 downloader + sequence/ground-truth loader
  detection/    YOLO wrapper + backbone specs
  tracking/     Kalman filter, IoU, Track lifecycle, ByteTrack, DeepSORT, ReID embedder
  evaluation/   py-motmetrics wrapper
  export/       ONNX export, INT8 quantization, latency/size/quality benchmark
  analytics/    trajectories, line-crossing counting, occupancy heatmap
  training/     detection-cache runner, Optuna tuning, full benchmark grid runner
dashboard/
  app.py        Dash app shell (navbar + page container)
  data_loader.py  shared read-only access to results/cache for every page
  pages/        one module per page (overview, backbone benchmark, tracker
                 comparison, sequence explorer, analytics)
tests/
  unit/         no external downloads, run in every CI job
  integration/  real YOLO11n checkpoint, isolated into its own CI job
notebooks/      EDA + figure generation only, no business logic
docs/           this document, methodology, experiments, limitations, results
```
