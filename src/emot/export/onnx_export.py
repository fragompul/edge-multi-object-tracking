"""YOLO -> ONNX export and the two INT8 quantisation flavours.

- **Dynamic** quantisation (weights INT8, activations quantised on the fly) needs no data
  but ONNX Runtime only has fast integer kernels for MatMul/Gemm: a convolutional
  detector gets little or no speed-up (profiled in `docs/edge.md`).
- **Static** (QDQ) quantisation calibrates activation ranges on a few real frames, so the
  whole conv graph runs as integer `QLinearConv` -- the variant that actually pays off on
  CPU.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_dynamic,
    quantize_static,
)
from ultralytics import YOLO

from emot.detection.onnx_detector import letterbox
from emot.detection.yolo import resolve_weights

logger = logging.getLogger(__name__)

# final 1x1 prediction convs of each Detect branch (box `cv2.k.2`, class `cv3.k.2`) and the DFL
_HEAD_CONV = re.compile(r"/cv[23]\.\d+/cv[23]\.\d+\.2/Conv$|/dfl/conv/Conv$")

_MODEL_PREFIX = re.compile(r"^/model\.(\d+)/")
_BRANCH = re.compile(r"/cv[23]\.")


def _decode_nodes(graph) -> list[str]:
    """Nodes of the Detect module that turn raw branch outputs into boxes and scores
    (DFL softmax, anchor arithmetic, sigmoid, concat). They mix value ranges of very
    different scale -- pixel coordinates up to ~640 next to probabilities in [0, 1] -- so
    one 8-bit scale per tensor flattens every score to zero. They are also cheap: always
    left in float."""
    last = max(
        (int(m.group(1)) for n in graph.node if (m := _MODEL_PREFIX.match(n.name))), default=-1
    )
    prefix = f"/model.{last}/"
    return [n.name for n in graph.node if n.name.startswith(prefix) and not _BRANCH.search(n.name)]


# 16:9 video at a given long side, padded up to the stride-32 grid (what Ultralytics'
# rectangular inference does): 640 -> 384x640, 1280 -> 736x1280.
def rect_shape_16x9(long_side: int) -> tuple[int, int]:
    height = int(np.ceil(long_side * 9 / 16 / 32) * 32)
    return height, long_side


def onnx_name(backbone_weights: str, shape_hw: tuple[int, int], variant: str = "fp32") -> str:
    stem = Path(backbone_weights).stem
    return f"{stem}_{shape_hw[0]}x{shape_hw[1]}.{variant}.onnx"


def export_onnx(weights: str, output_dir: Path, shape_hw: tuple[int, int] = (384, 640)) -> Path:
    """Static-shape FP32 ONNX (simplified, opset chosen by Ultralytics)."""
    dest = output_dir / onnx_name(weights, shape_hw)
    if dest.exists():
        return dest
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(resolve_weights(weights))
    exported = Path(model.export(format="onnx", imgsz=list(shape_hw), dynamic=False, simplify=True))
    exported.replace(dest)
    return dest


def quantize_onnx_dynamic(fp32_path: Path) -> Path:
    dest = fp32_path.with_name(fp32_path.name.replace(".fp32.", ".int8dyn."))
    if not dest.exists():
        quantize_dynamic(str(fp32_path), str(dest), weight_type=QuantType.QUInt8)
    return dest


class _FrameReader(CalibrationDataReader):
    def __init__(self, frames: Sequence[np.ndarray], input_name: str, shape_hw: tuple[int, int]):
        self._iter = iter(frames)
        self._name = input_name
        self._shape = shape_hw

    def get_next(self) -> dict[str, np.ndarray] | None:
        frame = next(self._iter, None)
        if frame is None:
            return None
        canvas, _ = letterbox(frame, self._shape)
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return {self._name: np.ascontiguousarray(tensor)}


def quantize_onnx_static(
    fp32_path: Path,
    calibration_images: Sequence[Path],
    shape_hw: tuple[int, int],
    keep_head_fp32: bool = False,
) -> Path:
    """QDQ static INT8 with per-channel INT8 weights and UINT8 activations, calibrated with
    MinMax ranges.

    `keep_head_fp32` leaves the box/class prediction convs and the DFL in float: the last
    layers map features to coordinates, where quantisation noise is least forgiving.
    """
    suffix = "int8sh" if keep_head_fp32 else "int8st"
    dest = fp32_path.with_name(fp32_path.name.replace(".fp32.", f".{suffix}."))
    if dest.exists():
        return dest
    pre = fp32_path.with_name(fp32_path.stem + ".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)

    import onnx

    graph = onnx.load(str(pre)).graph
    input_name = graph.input[0].name
    exclude = _decode_nodes(graph)
    if keep_head_fp32:
        exclude += [n.name for n in graph.node if n.op_type == "Conv" and _HEAD_CONV.search(n.name)]

    loaded = (cv2.imread(str(p)) for p in calibration_images)
    frames = [f for f in loaded if f is not None]
    quantize_static(
        str(pre),
        str(dest),
        _FrameReader(frames, input_name, shape_hw),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=sorted(set(exclude)),
    )
    pre.unlink(missing_ok=True)
    return dest
