"""Optuna hyper-parameter search for every tracker family, on a held-out
temporal validation split.

Protocol (and its honest limits -- see docs/limitations.md):
- One representative sequence/backbone (`MOT17-02` / `yolo11s` by default),
  tuned on its **trailing 25% of frames** and replayed from cached
  detections + embeddings, so a 40-trial study takes seconds.
- Objective is **HOTA**, not MOTA: MOTA barely registers identity errors,
  which is exactly what ReID and matching cascades are meant to fix, so a
  MOTA objective would tune them to look useless.
- BoT-SORT inherits ByteTrack's tuned association thresholds and tunes only
  its own appearance-fusion parameters, which keeps the CMC/ReID ablation
  clean: any gain is attributable to the mechanism, not to extra tuning.
- The tuned configs are then run on all 7 sequences, so how well a
  single-sequence config generalises is itself a reported result.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import optuna

from emot.benchmark.caches import CACHE_ROOT, CachedDetections, cmc_path, detection_path
from emot.benchmark.precompute import load_embeddings
from emot.benchmark.replay import VARIANTS, replay, score
from emot.data.mot17 import load_ground_truth, load_seqinfo, train_val_split

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ASSOCIATION = {
    "high_conf_threshold": (0.4, 0.8),
    "low_conf_threshold": (0.05, 0.3),
    "match_iou_threshold": (0.5, 0.9),
    "low_match_iou_threshold": (0.3, 0.7),
}


def _suggest_common(trial: optuna.Trial) -> dict:
    return {
        "min_hits": trial.suggest_int("min_hits", 1, 5),
        "max_age": trial.suggest_int("max_age", 10, 60),
    }


def _suggest(family: str, trial: optuna.Trial, bytetrack_params: dict) -> dict:
    if family == "bytetrack":
        params = {k: trial.suggest_float(k, *rng) for k, rng in ASSOCIATION.items()}
        return {**params, **_suggest_common(trial)}
    if family == "botsort":
        return {
            **bytetrack_params,
            "proximity_threshold": trial.suggest_float("proximity_threshold", 0.3, 0.8),
            "appearance_threshold": trial.suggest_float("appearance_threshold", 0.1, 0.4),
            "ema_alpha": trial.suggest_float("ema_alpha", 0.7, 0.98),
        }
    return {
        "max_cosine_distance": trial.suggest_float("max_cosine_distance", 0.05, 0.6),
        "iou_fallback_threshold": trial.suggest_float("iou_fallback_threshold", 0.2, 0.6),
        "min_confidence": trial.suggest_float("min_confidence", 0.2, 0.6),
        **_suggest_common(trial),
    }


def tune_variant(
    variant: str,
    sequence: str,
    backbone: str,
    data_root: Path,
    cache_root: Path,
    n_trials: int,
    bytetrack_params: dict,
    seed: int = 0,
) -> dict:
    spec = VARIANTS[variant]
    seq_dir = data_root / sequence
    _, val_frames = train_val_split(load_seqinfo(seq_dir))
    gt_df = load_ground_truth(seq_dir)
    cached = CachedDetections(detection_path(sequence, backbone, root=cache_root))
    emb = (
        load_embeddings(sequence, backbone, spec.embedder, cache_root=cache_root)
        if spec.embedder
        else None
    )
    warps = np.load(cmc_path(sequence, cache_root)) if spec.use_cmc else None

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(spec.family, trial, bytetrack_params)
        tracks = replay(
            spec, sequence, backbone, val_frames, {spec.tuned_key: params, "bytetrack": params},
            cache_root=cache_root, cached=cached, embeddings=emb, warps=warps,
        )
        return score(gt_df, tracks, val_frames).hota

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    logger.info("%-24s best val HOTA=%.4f", variant, study.best_value)
    params = dict(study.best_params)
    if spec.family == "botsort":
        params = {**bytetrack_params, **params}
    return {"best_hota": study.best_value, "params": params}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--sequence", default="MOT17-02")
    parser.add_argument("--backbone", default="yolo11s")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("results/tuned_tracker_configs.json"))
    args = parser.parse_args()

    out: dict = {
        "tuned_on_sequence": args.sequence,
        "tuned_on_backbone": args.backbone,
        "objective": "HOTA on the trailing 25% of frames",
        "n_trials": args.n_trials,
    }
    # (config key, variant used to tune it)
    out["bytetrack"] = tune_variant(
        "bytetrack", args.sequence, args.backbone, args.data_root, args.cache_dir,
        args.n_trials, {},
    )
    shared = (*ASSOCIATION, "min_hits", "max_age")
    bt = {k: v for k, v in out["bytetrack"]["params"].items() if k in shared}
    out["deepsort"] = tune_variant(
        "deepsort", args.sequence, args.backbone, args.data_root, args.cache_dir,
        args.n_trials, bt,
    )
    out["deepsort_osnet"] = tune_variant(
        "deepsort_osnet_cascade", args.sequence, args.backbone, args.data_root, args.cache_dir,
        args.n_trials, bt,
    )
    out["botsort"] = tune_variant(
        "botsort", args.sequence, args.backbone, args.data_root, args.cache_dir,
        args.n_trials, bt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
