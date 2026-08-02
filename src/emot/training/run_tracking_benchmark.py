"""Full (sequence x backbone x tracker) benchmark grid, using cached
detections (see `run_detection_cache.py`) and the tuned tracker configs
(see `tune_trackers.py`). Resumable: skips any
`results/<sequence>/<backbone>/<tracker>.json` that already exists.

Evaluated on the **full sequence** (train + val frames), not just val --
val was only held out for hyperparameter tuning; scoring the final
benchmark on the complete sequence uses all available ground truth,
matching standard MOT benchmark practice of reporting on whole sequences.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from emot.data.mot17 import iter_frame_indices, load_ground_truth, load_seqinfo
from emot.detection.yolo import BACKBONES
from emot.tracking.bytetrack import ByteTrackConfig
from emot.tracking.deepsort import DeepSortConfig
from emot.tracking.reid import AppearanceEmbedder
from emot.training.tracking_runner import load_cached_detections, run_bytetrack, run_deepsort

logger = logging.getLogger(__name__)


def _load_tuned_configs(path: Path) -> tuple[ByteTrackConfig, DeepSortConfig]:
    if not path.exists():
        logger.warning("No tuned config at %s, using defaults", path)
        return ByteTrackConfig(), DeepSortConfig()
    data = json.loads(path.read_text())
    return (
        ByteTrackConfig(**data["bytetrack"]["params"]),
        DeepSortConfig(**data["deepsort"]["params"]),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/detections"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--tuned-config", type=Path, default=Path("results/tuned_tracker_configs.json"))
    parser.add_argument(
        "--sequences", nargs="+", default=["MOT17-02", "MOT17-04", "MOT17-11", "MOT17-13"]
    )
    parser.add_argument("--backbones", nargs="+", default=[b.name for b in BACKBONES])
    args = parser.parse_args()

    bytetrack_config, deepsort_config = _load_tuned_configs(args.tuned_config)
    embedder = AppearanceEmbedder()

    for sequence in args.sequences:
        seq_dir = args.data_root / sequence
        seq_info = load_seqinfo(seq_dir)
        gt_df = load_ground_truth(seq_dir)
        frame_indices = iter_frame_indices(seq_info)

        for backbone_name in args.backbones:
            cache_path = args.cache_dir / sequence / f"{backbone_name}.npz"
            if not cache_path.exists():
                logger.info("SKIP (no detection cache yet): %s/%s", sequence, backbone_name)
                continue
            detections = load_cached_detections(cache_path)

            for tracker_name in ["bytetrack", "deepsort"]:
                result_path = args.results_dir / sequence / backbone_name / f"{tracker_name}.json"
                if result_path.exists():
                    logger.info("SKIP (cached): %s/%s/%s", sequence, backbone_name, tracker_name)
                    continue

                logger.info("Running: %s/%s/%s", sequence, backbone_name, tracker_name)
                start = time.time()
                if tracker_name == "bytetrack":
                    metrics = run_bytetrack(detections, gt_df, frame_indices, bytetrack_config)
                else:
                    metrics = run_deepsort(
                        detections, gt_df, frame_indices, deepsort_config, seq_dir, embedder
                    )
                elapsed = time.time() - start

                payload = {
                    "sequence": sequence,
                    "backbone": backbone_name,
                    "tracker": tracker_name,
                    "mota": metrics.mota,
                    "motp": metrics.motp,
                    "idf1": metrics.idf1,
                    "num_switches": metrics.num_switches,
                    "num_false_positives": metrics.num_false_positives,
                    "num_misses": metrics.num_misses,
                    "mostly_tracked": metrics.mostly_tracked,
                    "mostly_lost": metrics.mostly_lost,
                    "num_frames": metrics.num_frames,
                    "eval_time_sec": elapsed,
                }
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(payload, indent=2))
                logger.info(
                    "Done: %s/%s/%s -> mota=%.3f idf1=%.3f switches=%d",
                    sequence,
                    backbone_name,
                    tracker_name,
                    metrics.mota,
                    metrics.idf1,
                    metrics.num_switches,
                )


if __name__ == "__main__":
    main()
