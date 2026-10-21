"""Runs each YOLO backbone's detector once per sequence and caches the
raw per-frame detections to disk.

This exists because the tracking benchmark grid is (sequences x backbones
x trackers), but detection is the expensive step and doesn't depend on
which tracker is used downstream -- running it (sequences x backbones)
times and reusing the cache for every tracker avoids redoing the same
YOLO forward passes once per tracker for no benefit. Resumable: skips
any `(sequence, backbone, imgsz)` triple whose cache file already exists.

Detections are cached down to a low confidence floor (0.1) so that each
tracker can apply its *own* thresholds: ByteTrack's second association
stage genuinely needs the 0.1-0.4 band, DeepSORT wants a cleaner >=0.3
input. The latency recorded here is on whatever `--device` ran the
detection (Apple-silicon GPU by default) and is **not** an edge-latency
number -- those come from `run_export_benchmark.py`, on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from emot.benchmark.caches import (
    CACHE_ROOT,
    DEFAULT_IMGSZ,
    DETECTION_CONF_FLOOR,
    detection_path,
    run_key,
)
from emot.data.mot17 import iter_frame_indices, load_seqinfo
from emot.detection.yolo import BACKBONES, PersonDetector

logger = logging.getLogger(__name__)
ALL_SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_one(
    sequence: str, weights: str, data_root: Path, device: str, imgsz: int, conf: float
) -> dict[str, np.ndarray]:
    seq_dir = data_root / sequence
    seq_info = load_seqinfo(seq_dir)
    detector = PersonDetector(weights, confidence_threshold=conf, device=device, imgsz=imgsz)

    frame_ids, boxes, confs = [], [], []
    latencies_ms = []
    for frame_idx in iter_frame_indices(seq_info):
        frame_path = seq_dir / "img1" / f"{frame_idx:06d}.jpg"
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(f"could not read frame: {frame_path}")
        detections, elapsed = detector.detect(image)
        latencies_ms.append(elapsed * 1000)
        for det in detections:
            frame_ids.append(frame_idx)
            boxes.append(det.xyxy)
            confs.append(det.confidence)

    xyxy = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    return {
        "frame_id": np.array(frame_ids, dtype=np.int32),
        "x1": xyxy[:, 0],
        "y1": xyxy[:, 1],
        "x2": xyxy[:, 2],
        "y2": xyxy[:, 3],
        "conf": np.array(confs, dtype=np.float32),
        "mean_latency_ms": np.array([float(np.mean(latencies_ms[1:] or latencies_ms))]),
        "num_frames": np.array([seq_info.seq_length]),
        "imgsz": np.array([imgsz]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--sequences", nargs="+", default=ALL_SEQUENCES)
    parser.add_argument("--backbones", nargs="+", default=[b.name for b in BACKBONES])
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--conf", type=float, default=DETECTION_CONF_FLOOR)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    backbone_by_name = {b.name: b for b in BACKBONES}
    summary_path = args.cache_dir / "detections" / "summary.json"

    for sequence in args.sequences:
        for backbone_name in args.backbones:
            cache_path = detection_path(sequence, backbone_name, args.imgsz, args.cache_dir)
            if cache_path.exists():
                logger.info("SKIP (cached): %s / %s", sequence, run_key(backbone_name, args.imgsz))
                continue
            backbone = backbone_by_name[backbone_name]
            logger.info("Detecting: %s / %s on %s", sequence, run_key(backbone_name, args.imgsz),
                        args.device)
            start = time.time()
            arrays = run_one(
                sequence, backbone.weights, args.data_root, args.device, args.imgsz, args.conf
            )
            elapsed = time.time() - start
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **arrays)  # type: ignore[arg-type]

            # Written after every run, not only at the end: this loop takes a long
            # time and an interruption near the end must not lose the metadata.
            entry = {
                "num_detections": int(len(arrays["frame_id"])),
                "device": args.device,
                "mean_latency_ms": float(arrays["mean_latency_ms"][0]),
                "wall_time_sec": elapsed,
            }
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            summary[f"{sequence}/{run_key(backbone_name, args.imgsz)}"] = entry
            summary_path.write_text(json.dumps(summary, indent=2))
            logger.info("Done: %s / %s -> %s", sequence, run_key(backbone_name, args.imgsz), entry)


if __name__ == "__main__":
    main()
