"""Exports a YOLO checkpoint to ONNX and applies dynamic INT8
quantization, for the edge-deployment latency/size benchmark.

Dynamic (not static/calibrated) quantization is used deliberately: static
quantization needs a representative calibration dataset run through the
model to collect activation ranges, which is disproportionate setup for a
benchmark whose point is "how much do you get for free," not squeezing
out the last percent of accuracy -- documented as a scope trade-off in
docs/limitations.md, with the accuracy cost of that choice measured
directly (INT8 detection quality vs FP32, not assumed lossless).
"""

from __future__ import annotations

import logging
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def export_onnx(weights: str, output_dir: Path, image_size: int = 640) -> Path:
    """Exports `weights` (e.g. "yolo11n.pt") to a static-shape FP32 ONNX
    graph. Ultralytics writes the `.onnx` file next to the source
    weights by default; this moves it into `output_dir` for a
    predictable, per-benchmark-run layout."""
    model = YOLO(weights)
    exported_path = Path(
        model.export(format="onnx", imgsz=image_size, dynamic=False, simplify=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / exported_path.name
    if exported_path != dest:
        exported_path.replace(dest)
    return dest


def quantize_onnx_dynamic(fp32_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{fp32_path.stem}.int8.onnx"
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(dest),
        weight_type=QuantType.QUInt8,
    )
    return dest
