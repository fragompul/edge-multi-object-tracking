"""Runs each YOLO backbone's detector once per sequence and caches the
raw per-frame detections to disk.

This exists because the tracking benchmark grid is (sequences x backbones
x trackers), but detection is the expensive step and doesn't depend on
which tracker is used downstream -- running it (sequences x backbones)
times and reusing the cache for every tracker avoids redoing the same
YOLO forward passes 2x (once per tracker) for no benefit. Resumable: skips
any `(sequence, backbone)` pair whose cache file already exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from emot.data.mot17 import iter_frame_indices, load_seqinfo
from emot.detection.yolo import BACKBONES, PersonDetector

logger = logging.getLogger(__name__)


def _cache_path(cache_dir: Path, sequence: str, backbone_name: str) -> Path:
    return cache_dir / sequence / f"{backbone_name}.npz"


def run_one(sequence: str, backbone_weights: str, backbone_name: str, data_root: Path) -> dict:
    seq_dir = data_root / sequence
    seq_info = load_seqinfo(seq_dir)
    detector = PersonDetector(backbone_weights)

    frame_ids, x1s, y1s, x2s, y2s, confs = [], [], [], [], [], []
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
            x1s.append(det.xyxy[0])
            y1s.append(det.xyxy[1])
            x2s.append(det.xyxy[2])
            y2s.append(det.xyxy[3])
            confs.append(det.confidence)

    return {
        "frame_id": np.array(frame_ids, dtype=np.int32),
        "x1": np.array(x1s, dtype=np.float32),
        "y1": np.array(y1s, dtype=np.float32),
        "x2": np.array(x2s, dtype=np.float32),
        "y2": np.array(y2s, dtype=np.float32),
        "conf": np.array(confs, dtype=np.float32),
        "mean_latency_ms": np.array([float(np.mean(latencies_ms[1:] or latencies_ms))]),
        "num_frames": np.array([seq_info.seq_length]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/detections"))
    parser.add_argument(
        "--sequences", nargs="+", default=["MOT17-02", "MOT17-04", "MOT17-11", "MOT17-13"]
    )
    parser.add_argument("--backbones", nargs="+", default=[b.name for b in BACKBONES])
    args = parser.parse_args()

    backbone_by_name = {b.name: b for b in BACKBONES}
    summary_path = args.cache_dir / "summary.json"

    def _load_summary() -> dict:
        return json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for sequence in args.sequences:
        for backbone_name in args.backbones:
            cache_path = _cache_path(args.cache_dir, sequence, backbone_name)
            if cache_path.exists():
                logger.info("SKIP (cached): %s / %s", sequence, backbone_name)
                continue
            backbone = backbone_by_name[backbone_name]
            logger.info("Running detection: %s / %s", sequence, backbone_name)
            start = time.time()
            arrays = run_one(sequence, backbone.weights, backbone_name, args.data_root)
            elapsed = time.time() - start
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, **arrays)

            # Written after every (sequence, backbone) pair, not just once
            # at the end -- this loop runs for ~1-2 hours, and losing all
            # latency/count metadata to an interruption near the end would
            # be a needless resumability gap the per-file npz caches don't
            # otherwise have.
            entry = {
                "num_detections": int(len(arrays["frame_id"])),
                "mean_latency_ms": float(arrays["mean_latency_ms"][0]),
                "wall_time_sec": elapsed,
            }
            existing = _load_summary()
            existing[f"{sequence}/{backbone_name}"] = entry
            summary_path.write_text(json.dumps(existing, indent=2))
            logger.info("Done: %s / %s -> %s", sequence, backbone_name, entry)


if __name__ == "__main__":
    main()
