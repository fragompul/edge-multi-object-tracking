"""Builds the ReID-embedding and camera-motion caches from the detection
cache (run `run_detection_cache.py` first). Resumable."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from emot.benchmark.caches import CACHE_ROOT, DEFAULT_IMGSZ, detection_path
from emot.benchmark.precompute import build_cmc, build_embedder, build_embeddings
from emot.detection.yolo import BACKBONES
from emot.training.run_detection_cache import ALL_SEQUENCES, default_device

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)
    parser.add_argument("--sequences", nargs="+", default=ALL_SEQUENCES)
    parser.add_argument("--backbones", nargs="+", default=[b.name for b in BACKBONES])
    parser.add_argument("--embedders", nargs="+", default=["resnet18", "osnet_x0_25"])
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    embedders = {name: build_embedder(name, args.device) for name in args.embedders}
    for sequence in args.sequences:
        logger.info("camera motion: %s", sequence)
        build_cmc(sequence, args.data_root, args.cache_dir)
        for backbone in args.backbones:
            if not detection_path(sequence, backbone, args.imgsz, args.cache_dir).exists():
                logger.info("SKIP (no detections yet): %s/%s", sequence, backbone)
                continue
            logger.info("embeddings: %s/%s", sequence, backbone)
            build_embeddings(
                sequence, backbone, embedders, args.data_root, args.cache_dir, args.imgsz
            )


if __name__ == "__main__":
    main()
