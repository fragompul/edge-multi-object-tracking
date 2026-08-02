"""Shared read-only data access for every dashboard page: loads the
benchmark JSON results, cached detections, and MOT17 sequence metadata
straight from disk. Kept separate from the page modules so every page
reads results the same way, and so this logic is testable without
importing Dash at all.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from emot.data.mot17 import load_ground_truth, load_seqinfo


@lru_cache(maxsize=1)
def load_tracking_results(results_dir: Path) -> pd.DataFrame:
    """Flattens every `results/<sequence>/<backbone>/<tracker>.json` into
    one DataFrame -- the shape every comparison page needs."""
    rows = []
    for sequence_dir in sorted(results_dir.glob("*")):
        if not sequence_dir.is_dir():
            continue
        for backbone_dir in sorted(sequence_dir.glob("*")):
            if not backbone_dir.is_dir():
                continue
            for result_file in sorted(backbone_dir.glob("*.json")):
                rows.append(json.loads(result_file.read_text()))
    if not rows:
        return pd.DataFrame(
            columns=[
                "sequence",
                "backbone",
                "tracker",
                "mota",
                "motp",
                "idf1",
                "num_switches",
                "num_false_positives",
                "num_misses",
                "mostly_tracked",
                "mostly_lost",
                "num_frames",
                "eval_time_sec",
            ]
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_detection_summary(cache_dir: Path) -> dict:
    summary_path = cache_dir / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


@lru_cache(maxsize=1)
def load_export_benchmark(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "export_benchmark.json"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "backbone",
                "variant",
                "file_size_mb",
                "mean_latency_ms",
                "p95_latency_ms",
                "mean_detections_per_frame",
                "mean_iou_vs_reference",
            ]
        )
    return pd.DataFrame(json.loads(path.read_text()))


def available_sequences(data_root: Path) -> list[str]:
    return sorted(d.name for d in data_root.glob("MOT17-*") if d.is_dir())


def load_frame_detections(
    cache_dir: Path, sequence: str, backbone: str, frame_idx: int
) -> np.ndarray:
    """Returns an `(N, 5)` array of `[x1, y1, x2, y2, confidence]` for one
    cached frame -- used by the sequence explorer page to draw boxes
    without re-running the detector."""
    npz_path = cache_dir / sequence / f"{backbone}.npz"
    if not npz_path.exists():
        return np.zeros((0, 5))
    data = np.load(npz_path)
    mask = data["frame_id"] == frame_idx
    if not mask.any():
        return np.zeros((0, 5))
    return np.stack(
        [
            data["x1"][mask],
            data["y1"][mask],
            data["x2"][mask],
            data["y2"][mask],
            data["conf"][mask],
        ],
        axis=1,
    )


def load_sequence_ground_truth(data_root: Path, sequence: str):
    seq_dir = data_root / sequence
    return load_seqinfo(seq_dir), load_ground_truth(seq_dir)
