"""Loads the benchmark's JSON results into tidy DataFrames (one place, used by the CLI,
the dashboard, the figure scripts and the web-data exporter)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from emot.benchmark.caches import CACHE_ROOT, DEFAULT_IMGSZ, cmc_path
from emot.benchmark.replay import VARIANTS

SEQUENCES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
# per-frame camera translation (px) above which a sequence counts as "moving camera"
MOVING_CAMERA_PX = 1.0

METRIC_COLUMNS = ["mota", "idf1", "hota", "deta", "assa", "loca", "num_switches"]


def load_tracking_results(
    results_dir: Path = Path("results"), imgsz: int | None = DEFAULT_IMGSZ
) -> pd.DataFrame:
    """One row per (sequence, backbone, tracker). `imgsz=None` returns every input size
    (the resolution sweep); the default returns the main 640 px grid."""
    rows = []
    for path in sorted(results_dir.glob("MOT17-*/*/*.json")):
        if path.stem == "detection":
            continue
        rows.append(json.loads(path.read_text()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["label"] = df["tracker"].map(lambda t: VARIANTS[t].label if t in VARIANTS else t)
    return df if imgsz is None else df[df["imgsz"] == imgsz].reset_index(drop=True)


def load_detection_results(
    results_dir: Path = Path("results"), imgsz: int | None = DEFAULT_IMGSZ
) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(results_dir.glob("MOT17-*/*/detection.json"))]
    df = pd.DataFrame(rows)
    return df if df.empty or imgsz is None else df[df["imgsz"] == imgsz].reset_index(drop=True)


def load_edge_benchmark(results_dir: Path = Path("results")) -> dict:
    path = results_dir / "edge_benchmark.json"
    return json.loads(path.read_text()) if path.exists() else {"rows": [], "machine": {}}


def camera_motion_px(sequence: str, cache_root: Path = CACHE_ROOT) -> float | None:
    """Mean per-frame global translation estimated by CMC; ~0 for a tripod, >1 px hand-held."""
    path = cmc_path(sequence, cache_root)
    if not path.exists():
        return None
    warps = np.load(path)
    return float(np.linalg.norm(warps[1:, :, 2], axis=1).mean())


def sequence_table(results_dir: Path = Path("results")) -> pd.DataFrame:
    path = results_dir / "sequences.json"
    return pd.DataFrame(json.loads(path.read_text())) if path.exists() else pd.DataFrame()


def mean_by_tracker(df: pd.DataFrame, exclude_sequences: tuple[str, ...] = ()) -> pd.DataFrame:
    sub = df[~df["sequence"].isin(exclude_sequences)]
    return sub.groupby("tracker")[METRIC_COLUMNS].mean().sort_values("hota", ascending=False)
