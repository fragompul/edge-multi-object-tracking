"""Tracker variants and cache-driven replay.

A *variant* is one fully specified tracker configuration of the benchmark.
Replaying a variant over a sequence needs only cached arrays (detections,
ReID embeddings, camera-motion warps) -- no images, no neural nets -- so the
whole (sequence x backbone x variant) grid is cheap to re-run and to tune.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from emot.benchmark.caches import (
    CACHE_ROOT,
    DEFAULT_IMGSZ,
    CachedDetections,
    cmc_path,
    detection_path,
)
from emot.benchmark.precompute import load_embeddings
from emot.data.mot17 import SequenceInfo, gt_to_mot_array
from emot.evaluation.mot_metrics import MOTAccumulator, SequenceMetrics
from emot.tracking.botsort import BotSortConfig, BotSortTracker
from emot.tracking.bytetrack import ByteTrackConfig, ByteTracker
from emot.tracking.deepsort import DeepSortConfig, DeepSortTracker

# frame -> list of (track_id, xyxy, confidence)
FrameTracks = dict[int, list[tuple[int, np.ndarray, float]]]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    family: str  # "bytetrack" | "botsort" | "deepsort"
    embedder: str | None = None
    use_cmc: bool = False
    use_reid: bool = False
    cascade: bool = False
    tuned_key: str = "bytetrack"  # which entry of tuned_tracker_configs.json to start from


VARIANTS: dict[str, VariantSpec] = {
    v.name: v
    for v in [
        VariantSpec("bytetrack", "ByteTrack", "bytetrack"),
        VariantSpec("botsort_cmc", "BoT-SORT (CMC)", "botsort", use_cmc=True),
        VariantSpec(
            "botsort",
            "BoT-SORT (CMC + ReID)",
            "botsort",
            embedder="osnet_x0_25",
            use_cmc=True,
            use_reid=True,
        ),
        VariantSpec(
            "botsort_reid",
            "BoT-SORT (ReID only)",
            "botsort",
            embedder="osnet_x0_25",
            use_reid=True,
        ),
        VariantSpec(
            "deepsort",
            "DeepSORT (ResNet18)",
            "deepsort",
            embedder="resnet18",
            tuned_key="deepsort",
        ),
        VariantSpec(
            "deepsort_osnet",
            "DeepSORT (OSNet)",
            "deepsort",
            embedder="osnet_x0_25",
            tuned_key="deepsort_osnet",
        ),
        VariantSpec(
            "deepsort_osnet_cascade",
            "DeepSORT (OSNet + cascade)",
            "deepsort",
            embedder="osnet_x0_25",
            cascade=True,
            tuned_key="deepsort_osnet",
        ),
    ]
}
MAIN_VARIANTS = ["bytetrack", "deepsort", "deepsort_osnet_cascade", "botsort_cmc", "botsort"]
ABLATION_VARIANTS = ["botsort_reid", "deepsort_osnet"]


def _filter_kwargs(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return {k: v for k, v in params.items() if k in allowed}


def load_tuned(path: Path = Path("results/tuned_tracker_configs.json")) -> dict[str, dict]:
    """Tuned parameter dicts keyed by `tuned_key`; empty dict means 'use defaults'."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: v["params"] for k, v in data.items() if isinstance(v, dict) and "params" in v}


def build_tracker(spec: VariantSpec, tuned: dict[str, dict]) -> Any:
    params = tuned.get(spec.tuned_key, {})
    if spec.family == "bytetrack":
        return ByteTracker(ByteTrackConfig(**_filter_kwargs(ByteTrackConfig, params)))
    if spec.family == "botsort":
        base = tuned.get("bytetrack", {})
        cfg = BotSortConfig(**_filter_kwargs(BotSortConfig, base))
        cfg.use_cmc, cfg.use_reid = spec.use_cmc, spec.use_reid
        return BotSortTracker(cfg)
    dcfg = DeepSortConfig(**_filter_kwargs(DeepSortConfig, params))
    dcfg.cascade = spec.cascade
    return DeepSortTracker(dcfg)


def replay(
    spec: VariantSpec,
    sequence: str,
    backbone: str,
    frames: range,
    tuned: dict[str, dict],
    cache_root: Path = CACHE_ROOT,
    imgsz: int = DEFAULT_IMGSZ,
    cached: CachedDetections | None = None,
    embeddings: np.ndarray | None = None,
    warps: np.ndarray | None = None,
) -> FrameTracks:
    """Runs `spec` over cached inputs for `frames`; the optional arguments let
    callers (e.g. the Optuna loop) load the caches once and replay many times."""
    cached = cached or CachedDetections(detection_path(sequence, backbone, imgsz, cache_root))
    if spec.embedder and embeddings is None:
        embeddings = load_embeddings(sequence, backbone, spec.embedder, imgsz, cache_root)
    if spec.use_cmc and warps is None:
        warps = np.load(cmc_path(sequence, cache_root))

    tracker = build_tracker(spec, tuned)
    out: FrameTracks = {}
    for frame in frames:
        dets = cached.detections(frame)
        rows = cached.rows(frame)
        if spec.family == "bytetrack":
            tracks = tracker.update(dets)
        elif spec.family == "botsort":
            emb = embeddings[rows] if spec.use_reid and embeddings is not None else None
            warp = warps[frame - 1] if warps is not None else None
            tracks = tracker.update(dets, emb, warp)
        else:
            assert embeddings is not None
            tracks = tracker.update(dets, embeddings[rows])
        out[frame] = [(t.track_id, t.xyxy.copy(), float(t.confidence)) for t in tracks]
    return out


def score(gt_df: pd.DataFrame, tracks: FrameTracks, frames: range) -> SequenceMetrics:
    return score_with_events(gt_df, tracks, frames)[0]


def score_with_events(
    gt_df: pd.DataFrame, tracks: FrameTracks, frames: range
) -> tuple[SequenceMetrics, list[dict[str, int]]]:
    acc = MOTAccumulator()
    for frame in frames:
        rows = tracks.get(frame, [])
        acc.update_arrays(
            gt_to_mot_array(gt_df, frame),
            [r[0] for r in rows],
            np.array([r[1] for r in rows]) if rows else np.zeros((0, 4)),
        )
    metrics = acc.compute("seq")
    return metrics, acc.switch_events()


def sequence_frames(info: SequenceInfo) -> range:
    return range(1, info.seq_length + 1)
