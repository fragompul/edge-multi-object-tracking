"""Full (sequence x backbone x tracker-variant) benchmark grid plus
detection AP, replayed from caches (see `run_detection_cache.py` and
`build_feature_caches.py`) with the tuned configs (`tune_trackers.py`).

Resumable: skips any `results/<sequence>/<backbone>/<variant>.json` that
exists. Scored on the **full sequence** (val frames were held out only for
tuning), matching standard MOT practice of reporting whole sequences.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from emot.benchmark.caches import (
    CACHE_ROOT,
    DEFAULT_IMGSZ,
    CachedDetections,
    detection_path,
    run_key,
)
from emot.benchmark.detection_eval import evaluate_cached
from emot.benchmark.precompute import load_embeddings
from emot.benchmark.replay import (
    ABLATION_VARIANTS,
    MAIN_VARIANTS,
    VARIANTS,
    load_tuned,
    replay,
    score,
    sequence_frames,
)
from emot.data.mot17 import load_ground_truth, load_seqinfo
from emot.detection.yolo import BACKBONES
from emot.training.run_detection_cache import ALL_SEQUENCES

logger = logging.getLogger(__name__)


def run_grid(
    sequences: list[str],
    backbones: list[str],
    variants: list[str],
    data_root: Path,
    cache_root: Path,
    results_dir: Path,
    tuned_path: Path,
    imgsz: int = DEFAULT_IMGSZ,
) -> None:
    tuned = load_tuned(tuned_path)
    if not tuned:
        logger.warning("no tuned configs at %s -- using defaults", tuned_path)
    for sequence in sequences:
        seq_dir = data_root / sequence
        info = load_seqinfo(seq_dir)
        gt_df = load_ground_truth(seq_dir)
        frames = sequence_frames(info)
        for backbone in backbones:
            path = detection_path(sequence, backbone, imgsz, cache_root)
            if not path.exists():
                logger.info("SKIP (no detection cache): %s/%s", sequence, backbone)
                continue
            key = run_key(backbone, imgsz)
            out_dir = results_dir / sequence / key
            cached = CachedDetections(path)

            det_path = out_dir / "detection.json"
            if not det_path.exists():
                ap = evaluate_cached(cached, gt_df)
                out_dir.mkdir(parents=True, exist_ok=True)
                det_path.write_text(json.dumps({
                    "sequence": sequence, "backbone": backbone, "imgsz": imgsz,
                    "ap50": ap.ap50, "ap": ap.ap, "recall50": ap.recall50,
                    "num_gt": ap.num_gt, "num_detections": ap.num_detections,
                }, indent=2))
                logger.info("%s/%s detection AP50=%.3f AP=%.3f", sequence, key, ap.ap50, ap.ap)

            for name in variants:
                result_path = out_dir / f"{name}.json"
                if result_path.exists():
                    continue
                spec = VARIANTS[name]
                emb = (
                    load_embeddings(sequence, backbone, spec.embedder, imgsz, cache_root)
                    if spec.embedder else None
                )
                start = time.perf_counter()
                tracks = replay(
                    spec, sequence, backbone, frames, tuned,
                    cache_root=cache_root, imgsz=imgsz, cached=cached, embeddings=emb,
                )
                elapsed = time.perf_counter() - start
                m = score(gt_df, tracks, frames)
                payload = {
                    "sequence": sequence, "backbone": backbone, "imgsz": imgsz, "tracker": name,
                    "mota": m.mota, "motp": m.motp, "idf1": m.idf1,
                    "hota": m.hota, "deta": m.deta, "assa": m.assa, "loca": m.loca,
                    "num_switches": m.num_switches, "num_false_positives": m.num_false_positives,
                    "num_misses": m.num_misses, "mostly_tracked": m.mostly_tracked,
                    "mostly_lost": m.mostly_lost, "num_frames": m.num_frames,
                    "tracker_ms_per_frame": 1000 * elapsed / max(1, len(frames)),
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(payload, indent=2))
                logger.info(
                    "%s/%s/%-22s MOTA=%.3f IDF1=%.3f HOTA=%.3f IDsw=%d",
                    sequence, key, name, m.mota, m.idf1, m.hota, m.num_switches,
                )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--tuned-config", type=Path, default=Path("results/tuned_tracker_configs.json")
    )
    parser.add_argument("--sequences", nargs="+", default=ALL_SEQUENCES)
    parser.add_argument("--backbones", nargs="+", default=[b.name for b in BACKBONES])
    parser.add_argument("--variants", nargs="+", default=[*MAIN_VARIANTS, *ABLATION_VARIANTS])
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    args = parser.parse_args()
    run_grid(
        args.sequences, args.backbones, args.variants, args.data_root, args.cache_dir,
        args.results_dir, args.tuned_config, args.imgsz,
    )


if __name__ == "__main__":
    main()
