"""Generates the cross-language parity fixtures used by the web app's tests.

The browser tracker, analytics tools and YOLO post-processing are TypeScript ports of the
Python implementations. This script runs the *Python* code on fixed inputs and records
inputs + outputs, so `npm test` can assert that the ports agree numerically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from emot.agent.planner import ask_offline
from emot.agent.tools import TrackStore
from emot.benchmark.caches import CachedDetections, detection_path
from emot.detection.onnx_detector import decode_person_boxes, letterbox, nms_xyxy
from emot.detection.yolo import Detection
from emot.tracking.bytetrack import ByteTrackConfig, ByteTracker

OUT = Path("web/src/core/__fixtures__")
N_FRAMES = 150


def tracker_fixture() -> dict:
    cached = CachedDetections(detection_path("MOT17-02", "yolo11s"))
    cfg = ByteTrackConfig()
    tracker = ByteTracker(cfg)
    frames_in, frames_out = [], []
    for frame in range(1, N_FRAMES + 1):
        rows = cached.rows(frame)
        dets = [
            Detection(np.round(cached.boxes[i], 3), float(np.round(cached.conf[i], 4)))
            for i in rows
        ]
        frames_in.append([[*map(float, d.xyxy), d.confidence] for d in dets])
        tracks = tracker.update(dets)
        frames_out.append([[t.track_id, *map(float, t.xyxy), t.confidence] for t in tracks])
    return {
        "config": cfg.__dict__,
        "detections": frames_in,
        "tracks": frames_out,
    }


def analytics_fixture(tracks: list[list[list[float]]]) -> dict:
    frames = {i + 1: {int(t[0]): np.array(t[1:5]) for t in frame} for i, frame in enumerate(tracks)}
    store = TrackStore(frames, fps=30.0, width=1920, height=1080)
    calls = [
        ("scene_summary", {}),
        ("count_people", {}),
        ("count_people", {"region": "left", "start_s": 1, "end_s": 4}),
        ("line_crossings", {"x1": 0.5, "y1": 0, "x2": 0.5, "y2": 1}),
        ("line_crossings", {"x1": 0, "y1": 0.6, "x2": 1, "y2": 0.6, "start_s": 0.5}),
        ("dwell_time", {"region": "center", "top_k": 3}),
        ("crowding", {"threshold": 5, "min_seconds": 0.5}),
        ("busiest_moments", {"top_k": 3}),
        ("track_info", {"track_id": int(tracks[60][0][0])}),
    ]
    questions = [
        "How many people are there?",
        "How many people crossed the vertical line?",
        "How long did people stay in the center?",
        "When was it busiest?",
        "how many people in the last 2 seconds",
    ]
    return {
        "calls": [{"name": n, "args": a, "result": store.call(n, a)} for n, a in calls],
        "questions": [
            {
                "q": q,
                "text": (r := ask_offline(store, q)).text,
                "tools": [c.name for c in r.tool_calls],
            }
            for q in questions
        ],
    }


def yolo_fixture() -> dict:
    rng = np.random.default_rng(0)
    anchors = 600
    raw = np.zeros((1, 84, anchors), dtype=np.float32)
    raw[0, 0] = rng.uniform(50, 600, anchors)
    raw[0, 1] = rng.uniform(50, 330, anchors)
    raw[0, 2] = rng.uniform(15, 60, anchors)
    raw[0, 3] = rng.uniform(40, 140, anchors)
    raw[0, 4] = rng.uniform(0, 1, anchors) ** 3
    raw = raw.round(5)  # the fixture stores 5 decimals; decode exactly what the test will read
    boxes_c, scores = decode_person_boxes(raw, 0.1)
    cx, cy, w, h = boxes_c.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep = nms_xyxy(xyxy, scores, 0.7)
    geoms = []
    for w_img, h_img, th, tw in [
        (1920, 1080, 384, 640),
        (640, 480, 384, 640),
        (1280, 720, 736, 1280),
    ]:
        _, lb = letterbox(np.zeros((h_img, w_img, 3), np.uint8), (th, tw))
        geoms.append(
            {
                "w": w_img,
                "h": h_img,
                "th": th,
                "tw": tw,
                "scale": lb.scale,
                "pad_x": lb.pad_x,
                "pad_y": lb.pad_y,
            }
        )
    return {
        "anchors": anchors,
        "raw": raw[0].ravel().round(5).tolist(),
        "n_decoded": int(len(scores)),
        "kept": [int(i) for i in keep],
        "geometries": geoms,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tracker = tracker_fixture()
    (OUT / "bytetrack.json").write_text(json.dumps(tracker))
    (OUT / "analytics.json").write_text(json.dumps(analytics_fixture(tracker["tracks"])))
    (OUT / "yolo.json").write_text(json.dumps(yolo_fixture()))
    print("wrote fixtures to", OUT)


if __name__ == "__main__":
    main()
