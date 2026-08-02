"""Loader for a MOT17 sequence in the canonical MOTChallenge directory
convention: `img1/<frame>.jpg`, `gt/gt.txt`, `seqinfo.ini`.

`gt.txt` columns (MOT17 ground-truth format, 1-indexed frames and IDs):
    frame, id, bb_left, bb_top, bb_width, bb_height, confidence, class, visibility

`class == 1` is "pedestrian" (the only class this project tracks --
MOT17 also labels static people, occluded people, reflections, etc. under
other class ids, which are excluded here since they are explicitly
non-target by the benchmark's own convention).
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PEDESTRIAN_CLASS = 1
GT_COLUMNS = [
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "confidence",
    "class",
    "visibility",
]


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    frame_rate: int
    seq_length: int
    im_width: int
    im_height: int


def load_seqinfo(seq_dir: Path) -> SequenceInfo:
    config = configparser.ConfigParser()
    config.read(seq_dir / "seqinfo.ini")
    section = config["Sequence"]
    return SequenceInfo(
        name=section["name"],
        frame_rate=int(section["frameRate"]),
        seq_length=int(section["seqLength"]),
        im_width=int(section["imWidth"]),
        im_height=int(section["imHeight"]),
    )


def load_ground_truth(seq_dir: Path, min_visibility: float = 0.0) -> pd.DataFrame:
    """Loads `gt/gt.txt`, filtered to visible pedestrians with a valid flag.

    The 7th column (`confidence`, called `consider_entry_flag` for ground
    truth) marks whether an annotated box should count towards evaluation
    at all -- MOT17 sets this to 0 for boxes that exist only to explain
    occlusions but shouldn't be scored.
    """
    gt_path = seq_dir / "gt" / "gt.txt"
    df = pd.read_csv(gt_path, header=None, names=GT_COLUMNS)
    df = df[
        (df["class"] == PEDESTRIAN_CLASS)
        & (df["confidence"] == 1)
        & (df["visibility"] >= min_visibility)
    ].copy()
    df["bb_right"] = df["bb_left"] + df["bb_width"]
    df["bb_bottom"] = df["bb_top"] + df["bb_height"]
    return df.reset_index(drop=True)


def frame_path(seq_dir: Path, frame_idx: int) -> Path:
    return seq_dir / "img1" / f"{frame_idx:06d}.jpg"


def iter_frame_indices(seq_info: SequenceInfo) -> range:
    return range(1, seq_info.seq_length + 1)


def train_val_split(seq_info: SequenceInfo, val_fraction: float = 0.25) -> tuple[range, range]:
    """Temporal split (never random) -- val is the trailing fraction of
    frames, since a random per-frame split would leak future/past
    context for the *same track* across train and val and make tracker
    hyperparameter tuning look better than it really is.
    """
    n_val = max(1, int(round(seq_info.seq_length * val_fraction)))
    n_train = seq_info.seq_length - n_val
    return range(1, n_train + 1), range(n_train + 1, seq_info.seq_length + 1)


def gt_to_mot_array(df: pd.DataFrame, frame_idx: int) -> np.ndarray:
    """Returns an (N, 5) array of [id, bb_left, bb_top, bb_width, bb_height]
    for one frame, the shape `motmetrics` accumulators expect."""
    rows = df[df["frame"] == frame_idx]
    return rows[["id", "bb_left", "bb_top", "bb_width", "bb_height"]].to_numpy(dtype=float)
