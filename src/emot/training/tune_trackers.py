"""Optuna hyperparameter search for ByteTrack/DeepSORT association
thresholds, optimizing MOTA on a held-out temporal val split.

Tuned on a **single representative sequence** (`MOT17-02` by default,
whichever cached-detection sequence finishes first in the detection-cache
run) rather than all 4, to bound compute -- this is a scope trade-off
(see docs/limitations.md), and its cost is measured directly: the tuned
config is then evaluated on the *other 3* sequences too, so how well a
single-sequence-tuned config generalizes is an honest, reported result,
not an assumption.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import optuna

from emot.data.mot17 import load_ground_truth, load_seqinfo, train_val_split
from emot.tracking.bytetrack import ByteTrackConfig
from emot.tracking.deepsort import DeepSortConfig
from emot.tracking.reid import AppearanceEmbedder
from emot.training.tracking_runner import load_cached_detections, run_bytetrack, run_deepsort

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _bytetrack_objective(trial: optuna.Trial, detections, gt_df, val_frames) -> float:
    config = ByteTrackConfig(
        high_conf_threshold=trial.suggest_float("high_conf_threshold", 0.4, 0.8),
        low_conf_threshold=trial.suggest_float("low_conf_threshold", 0.05, 0.3),
        match_iou_threshold=trial.suggest_float("match_iou_threshold", 0.5, 0.9),
        low_match_iou_threshold=trial.suggest_float("low_match_iou_threshold", 0.3, 0.7),
        min_hits=trial.suggest_int("min_hits", 1, 5),
        max_age=trial.suggest_int("max_age", 10, 60),
    )
    metrics = run_bytetrack(detections, gt_df, val_frames, config)
    return metrics.mota


def _deepsort_objective(
    trial: optuna.Trial, detections, gt_df, val_frames, seq_dir, embedder
) -> float:
    config = DeepSortConfig(
        max_cosine_distance=trial.suggest_float("max_cosine_distance", 0.05, 0.5),
        iou_fallback_threshold=trial.suggest_float("iou_fallback_threshold", 0.2, 0.6),
        min_hits=trial.suggest_int("min_hits", 1, 5),
        max_age=trial.suggest_int("max_age", 10, 60),
    )
    metrics = run_deepsort(detections, gt_df, val_frames, config, seq_dir, embedder)
    return metrics.mota


def tune(
    sequence: str,
    backbone_name: str,
    data_root: Path,
    cache_dir: Path,
    n_trials: int,
) -> dict:
    seq_dir = data_root / sequence
    seq_info = load_seqinfo(seq_dir)
    gt_df = load_ground_truth(seq_dir)
    _, val_frames = train_val_split(seq_info)

    detections = load_cached_detections(cache_dir / sequence / f"{backbone_name}.npz")

    logger.info(
        "Tuning ByteTrack on %s/%s (%d val frames)...", sequence, backbone_name, len(val_frames)
    )
    bt_study = optuna.create_study(direction="maximize")
    bt_study.optimize(
        lambda trial: _bytetrack_objective(trial, detections, gt_df, val_frames),
        n_trials=n_trials,
    )
    logger.info("ByteTrack best MOTA=%.4f params=%s", bt_study.best_value, bt_study.best_params)

    logger.info("Tuning DeepSORT on %s/%s...", sequence, backbone_name)
    embedder = AppearanceEmbedder()
    ds_study = optuna.create_study(direction="maximize")
    ds_study.optimize(
        lambda trial: _deepsort_objective(trial, detections, gt_df, val_frames, seq_dir, embedder),
        n_trials=n_trials,
    )
    logger.info("DeepSORT best MOTA=%.4f params=%s", ds_study.best_value, ds_study.best_params)

    return {
        "tuned_on_sequence": sequence,
        "tuned_on_backbone": backbone_name,
        "n_trials": n_trials,
        "bytetrack": {"best_mota": bt_study.best_value, "params": bt_study.best_params},
        "deepsort": {"best_mota": ds_study.best_value, "params": ds_study.best_params},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/detections"))
    parser.add_argument("--sequence", default="MOT17-02")
    parser.add_argument("--backbone", default="yolo11s")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--output", type=Path, default=Path("results/tuned_tracker_configs.json"))
    args = parser.parse_args()

    result = tune(args.sequence, args.backbone, args.data_root, args.cache_dir, args.n_trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
