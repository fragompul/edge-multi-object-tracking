# CLI reference

```text
emot version
emot track SOURCE [--tracker T] [--model M | --onnx PATH --provider cpu|coreml|cuda --threads N]
                  [--imgsz N] [--conf C] [--device auto|cpu|mps|cuda]
                  [--save out.mp4] [--jsonl tracks.jsonl] [--show] [--max-frames N] [--trail N]
emot ask TRACKS.jsonl "question" [--offline] [--model claude-opus-5-5] [--fps F --width W --height H] [--trace]
emot results [--metric hota|idf1|mota|num_switches]
```

`SOURCE` is a video file, a webcam index, a network URL, or an image folder (MOTChallenge `img1/` works directly).

`emot track` prints throughput and a per-stage breakdown (detector vs tracker milliseconds per frame, frames shed on live sources).
`--jsonl` writes one line per frame: `{"frame": 12, "tracks": [{"id": 3, "xyxy": [..], "conf": 0.87}]}`.

Other entry points: `python src/emot/training/*.py` (benchmark stages), `scripts/*.py` (figures, GIFs, web data).
