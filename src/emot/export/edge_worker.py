"""One isolated measurement: load a detector, time it over a fixed frame sample, score its
boxes against MOT17 ground truth, and report peak memory.

Runs in its own process (`python -m emot.export.edge_worker spec.json`) so that peak-RSS
numbers belong to exactly one configuration and no model, allocator state or thermal
history leaks between measurements. Prints a single JSON line on stdout.
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from emot.benchmark.detection_eval import MIN_EVAL_VISIBILITY
from emot.data.mot17 import load_ground_truth
from emot.evaluation.detection_ap import evaluate_detections


def peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if platform.system() == "Darwin" else raw / 1024


def build_detector(spec: dict):
    conf = spec.get("conf", 0.1)
    if spec["kind"] == "torch":
        from emot.detection.yolo import PersonDetector

        return PersonDetector(spec["weights"], conf, spec["device"], spec.get("imgsz", 640))
    from emot.detection.onnx_detector import OnnxPersonDetector

    return OnnxPersonDetector(
        spec["model"],
        conf,
        provider=spec.get("provider", "cpu"),
        num_threads=spec.get("threads", 0),
    )


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    data_root = Path(spec["data_root"])

    t0 = time.perf_counter()
    detector = build_detector(spec)
    first = cv2.imread(
        str(data_root / spec["frames"][0][0] / "img1" / f"{spec['frames'][0][1]:06d}.jpg")
    )
    for _ in range(spec.get("warmup", 5)):
        detector.detect(first)
    load_s = time.perf_counter() - t0

    gt_cache: dict[str, object] = {}
    det_by_frame: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    gt_by_frame: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    latencies = []
    for seq, frame in spec["frames"]:
        image = cv2.imread(str(data_root / seq / "img1" / f"{frame:06d}.jpg"))
        dets, elapsed = detector.detect(image)
        latencies.append(elapsed * 1000)
        key = (seq, frame)
        det_by_frame[key] = (
            np.array([d.xyxy for d in dets]).reshape(-1, 4),
            np.array([d.confidence for d in dets]),
        )
        if seq not in gt_cache:
            gt_cache[seq] = load_ground_truth(data_root / seq)
        rows = gt_cache[seq][gt_cache[seq]["frame"] == frame]  # type: ignore[index]
        gt_by_frame[key] = (
            rows[["bb_left", "bb_top", "bb_right", "bb_bottom"]].to_numpy(float),
            (rows["visibility"] < MIN_EVAL_VISIBILITY).to_numpy(),
        )

    # frames from different sequences must not share an index space: remap keys to ints
    index = {k: i for i, k in enumerate(det_by_frame)}
    ap = evaluate_detections(
        {index[k]: v for k, v in det_by_frame.items()},
        {index[k]: v for k, v in gt_by_frame.items()},
    )
    lat = np.array(latencies)
    print(
        json.dumps(
            {
                "mean_ms": float(lat.mean()),
                "median_ms": float(np.median(lat)),
                "p95_ms": float(np.percentile(lat, 95)),
                "fps": float(1000 / lat.mean()),
                "peak_rss_mb": peak_rss_mb(),
                "load_s": load_s,
                "ap50": ap.ap50,
                "ap": ap.ap,
                "recall50": ap.recall50,
                "detections_per_frame": ap.num_detections / len(latencies),
                "frames": len(latencies),
                "active_providers": getattr(detector, "active_providers", None),
            }
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
