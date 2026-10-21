"""Shared read-only data access for every dashboard page: benchmark results, caches and MOT17
sequences straight from disk. Kept separate from the pages so every page reads results the same
way, and so this logic is testable without importing Dash."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from emot.benchmark import aggregate
from emot.benchmark.caches import CachedDetections, detection_path
from emot.benchmark.replay import VARIANTS, FrameTracks, load_tuned, replay, sequence_frames
from emot.data.mot17 import load_ground_truth, load_seqinfo


@lru_cache(maxsize=1)
def load_tracking_results(results_dir: Path) -> pd.DataFrame:
    """One row per (sequence, backbone, tracker) of the main 640 px grid, with a `label` column."""
    return aggregate.load_tracking_results(results_dir)


@lru_cache(maxsize=1)
def load_detection_results(results_dir: Path) -> pd.DataFrame:
    return aggregate.load_detection_results(results_dir)


@lru_cache(maxsize=1)
def load_sequence_table(results_dir: Path) -> pd.DataFrame:
    return aggregate.sequence_table(results_dir)


@lru_cache(maxsize=1)
def load_edge_rows(results_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(aggregate.load_edge_benchmark(results_dir)["rows"])


@lru_cache(maxsize=1)
def load_detection_summary(cache_dir: Path) -> dict:
    summary_path = cache_dir / "summary.json"
    return json.loads(summary_path.read_text()) if summary_path.exists() else {}


def available_sequences(data_root: Path) -> list[str]:
    return sorted(d.name for d in data_root.glob("MOT17-*") if d.is_dir())


@lru_cache(maxsize=16)
def tracks_for(
    sequence: str,
    backbone: str,
    tracker: str,
    data_root: Path = Path("data/mot17"),
    cache_root: Path = Path("cache"),
) -> FrameTracks:
    """Replays a tracker variant over the cached detections of a sequence (seconds, no GPU)."""
    info = load_seqinfo(data_root / sequence)
    return replay(
        VARIANTS[tracker],
        sequence,
        backbone,
        sequence_frames(info),
        load_tuned(),
        cache_root=cache_root,
    )


def load_frame_detections(
    cache_dir: Path, sequence: str, backbone: str, frame_idx: int
) -> np.ndarray:
    """`(N, 5)` array of `[x1, y1, x2, y2, confidence]` for one cached frame."""
    path = detection_path(sequence, backbone, root=cache_dir.parent)
    if not path.exists():
        return np.zeros((0, 5))
    cached = CachedDetections(path)
    rows = cached.rows(frame_idx)
    return (
        np.column_stack([cached.boxes[rows], cached.conf[rows]]) if len(rows) else np.zeros((0, 5))
    )


def load_sequence_ground_truth(data_root: Path, sequence: str):
    seq_dir = data_root / sequence
    return load_seqinfo(seq_dir), load_ground_truth(seq_dir)
