"""Exports each YOLO backbone to ONNX + INT8, benchmarks all three
variants (PyTorch FP32, ONNX FP32, ONNX INT8) on a fixed sample of real
MOT17 frames, and writes `results/export_benchmark.json` -- the data the
dashboard's Backbone Benchmark page reads for the edge-export comparison.

Resumable at the backbone level: skips any backbone whose three variants
are already present in the output file.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import cv2

from emot.detection.yolo import BACKBONES
from emot.export.benchmark import benchmark_variant
from emot.export.onnx_export import export_onnx, quantize_onnx_dynamic

logger = logging.getLogger(__name__)


def _sample_frames(data_root: Path, sequence: str, n_frames: int) -> list:
    img_dir = data_root / sequence / "img1"
    paths = sorted(img_dir.glob("*.jpg"))[:n_frames]
    images = [cv2.imread(str(p)) for p in paths]
    return [img for img in images if img is not None]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mot17"))
    parser.add_argument("--sequence", default="MOT17-02")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--export-dir", type=Path, default=Path("exports"))
    parser.add_argument("--output", type=Path, default=Path("results/export_benchmark.json"))
    parser.add_argument("--backbones", nargs="+", default=[b.name for b in BACKBONES])
    args = parser.parse_args()

    backbone_by_name = {b.name: b for b in BACKBONES}
    images = _sample_frames(args.data_root, args.sequence, args.n_frames)
    logger.info("Benchmarking on %d real frames from %s", len(images), args.sequence)

    existing_rows = (
        json.loads(args.output.read_text()) if args.output.exists() else []
    )
    done_backbones = {row["backbone"] for row in existing_rows}
    rows = list(existing_rows)

    for backbone_name in args.backbones:
        if backbone_name in done_backbones:
            logger.info("SKIP (cached): %s", backbone_name)
            continue
        backbone = backbone_by_name[backbone_name]
        logger.info("Exporting + benchmarking: %s", backbone_name)

        fp32_onnx = export_onnx(backbone.weights, args.export_dir / backbone_name)
        int8_onnx = quantize_onnx_dynamic(fp32_onnx, args.export_dir / backbone_name)

        pt_result, pt_detections = benchmark_variant(
            "pytorch_fp32", Path(backbone.weights), images
        )
        onnx_result, _ = benchmark_variant(
            "onnx_fp32", fp32_onnx, images, reference_detections=pt_detections
        )
        int8_result, _ = benchmark_variant(
            "onnx_int8", int8_onnx, images, reference_detections=pt_detections
        )

        for result in [pt_result, onnx_result, int8_result]:
            row = {"backbone": backbone_name, **asdict(result)}
            rows.append(row)
            logger.info("Done: %s", row)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
