"""Dependency-light YOLO person detector on ONNX Runtime.

Why not just call Ultralytics on the `.onnx` file? Because the edge story
needs full control of the three things that decide real-world latency and
that Ultralytics hides: the **execution provider** (CPU / CoreML / CUDA),
the **thread budget**, and the **pre/post-processing** (letterbox, decode,
NMS) -- all of which also have to be re-implemented bit-for-bit in
TypeScript for the in-browser demo. This module is the Python reference
for that port (see `web/src/inference/yolo.ts`).

The input is a *static* `(H, W)` tensor. Match it to the camera's aspect
ratio (e.g. 384x640 for 16:9 video): padding a 16:9 frame into a 640x640
square wastes ~40% of the compute on grey bars, which is exactly the
mistake that made the first version of this project's ONNX export look
slower than PyTorch (see docs/edge.md).

Supports the standard Ultralytics detect head export: one output of shape
`(1, 4 + num_classes, num_anchors)` holding `[cx, cy, w, h, class scores...]`
in letterboxed input pixels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from emot.detection.yolo import COCO_PERSON_CLASS_ID, Detection

LETTERBOX_PAD_VALUE = 114


@dataclass(frozen=True)
class Letterbox:
    scale: float
    pad_x: float
    pad_y: float


def letterbox(image_bgr: np.ndarray, shape_hw: tuple[int, int]) -> tuple[np.ndarray, Letterbox]:
    """Aspect-preserving resize into a `(H, W)` canvas padded with grey 114
    (the Ultralytics convention), centred."""
    target_h, target_w = shape_hw
    h, w = image_bgr.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (target_w - new_w) / 2, (target_h - new_h) / 2
    canvas = np.full((target_h, target_w, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas, Letterbox(scale=scale, pad_x=float(left), pad_y=float(top))


def decode_person_boxes(
    raw: np.ndarray, conf_threshold: float, person_class: int = COCO_PERSON_CLASS_ID
) -> tuple[np.ndarray, np.ndarray]:
    """`raw` is `(1, 4+C, A)` (or its transpose). Returns `(boxes_cxcywh, scores)`
    for anchors whose person score clears the threshold."""
    pred = raw[0]
    if pred.shape[0] > pred.shape[1]:  # (A, 4+C) layout
        pred = pred.T
    scores = pred[4 + person_class]
    keep = scores >= conf_threshold
    return pred[:4, keep].T, scores[keep]


def nms_xyxy(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_det: int = 300
) -> np.ndarray:
    """Greedy NMS; returns kept indices ordered by descending score."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    xywh = np.stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1
    )
    # top_k=0: OpenCV's top_k would truncate the *candidates before* suppression, unlike
    # Ultralytics, which suppresses first and then caps the survivors at max_det.
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou_threshold)
    return np.asarray(keep, dtype=np.int64).reshape(-1)[:max_det]


def providers_for(name: str) -> list[str]:
    table = {
        "cpu": ["CPUExecutionProvider"],
        "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    if name not in table:
        raise ValueError(f"provider must be one of {sorted(table)}")
    return table[name]


class OnnxPersonDetector:
    """Same `detect()` contract as `PersonDetector`, running a static-shape
    Ultralytics ONNX export through ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        provider: str = "cpu",
        num_threads: int = 0,
    ):
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads  # 0 = ORT default (all cores)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers_for(provider)
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.active_providers = self.session.get_providers()

    def detect(self, image_bgr: np.ndarray) -> tuple[list[Detection], float]:
        start = time.perf_counter()
        canvas, lb = letterbox(image_bgr, self.input_hw)
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        raw = self.session.run(None, {self.input_name: np.ascontiguousarray(tensor)})[0]
        boxes_c, scores = decode_person_boxes(raw, self.confidence_threshold)
        if len(scores):
            cx, cy, w, h = boxes_c.T
            xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            keep = nms_xyxy(xyxy, scores, self.iou_threshold)
            xyxy, scores = xyxy[keep], scores[keep]
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - lb.pad_x) / lb.scale
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - lb.pad_y) / lb.scale
            h_img, w_img = image_bgr.shape[:2]
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w_img)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h_img)
        else:
            xyxy = np.zeros((0, 4))
        elapsed = time.perf_counter() - start
        detections = [
            Detection(xyxy=box.astype(float), confidence=float(score))
            for box, score in zip(xyxy, scores, strict=True)
        ]
        return detections, elapsed
