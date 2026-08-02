"""Thin wrapper around Ultralytics YOLO for person detection.

Kept as a wrapper (not used inline) so tracking/evaluation code depends on
a small, stable `Detection` interface rather than the `ultralytics`
result object shape directly -- useful both for testing (a fake detector
can implement the same interface) and if the underlying detector library
ever changes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

COCO_PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    """One detected box in a single frame, in `[left, top, right, bottom]`
    pixel coordinates."""

    xyxy: np.ndarray  # shape (4,)
    confidence: float


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    weights: str  # ultralytics model id, e.g. "yolo11n.pt"
    generation: str  # "yolov8" | "yolo11"
    size: str  # "n" | "s" | "m"


BACKBONES: list[BackboneSpec] = [
    BackboneSpec("yolov8n", "yolov8n.pt", "yolov8", "n"),
    BackboneSpec("yolov8s", "yolov8s.pt", "yolov8", "s"),
    BackboneSpec("yolov8m", "yolov8m.pt", "yolov8", "m"),
    BackboneSpec("yolo11n", "yolo11n.pt", "yolo11", "n"),
    BackboneSpec("yolo11s", "yolo11s.pt", "yolo11", "s"),
    BackboneSpec("yolo11m", "yolo11m.pt", "yolo11", "m"),
]


class PersonDetector:
    """Wraps a single Ultralytics YOLO checkpoint, filtered to the COCO
    `person` class (id 0) since MOT17 is a pedestrian-tracking benchmark."""

    def __init__(self, weights: str, confidence_threshold: float = 0.25, device: str = "cpu"):
        self.model = YOLO(weights)
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect(self, image: np.ndarray) -> tuple[list[Detection], float]:
        """`image` is a BGR array (as read by OpenCV). Returns detections
        plus the wall-clock inference time in seconds, for latency
        benchmarking."""
        start = time.perf_counter()
        results = list(
            self.model.predict(
                image,
                classes=[COCO_PERSON_CLASS_ID],
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
        )
        elapsed = time.perf_counter() - start

        detections = []
        result = results[0]
        assert isinstance(result, Results)
        boxes = result.boxes
        assert boxes is not None
        # ultralytics' stubs type `Boxes.xyxy`/`.conf` as `Tensor | ndarray` (the ndarray
        # variant only applies in ONNX/TensorRT streaming modes this project doesn't use);
        # always a Tensor here, hence the targeted ignore rather than a broad cast.
        xyxy_arr = boxes.xyxy.cpu().numpy()  # type: ignore[union-attr]
        conf_arr = boxes.conf.cpu().numpy()  # type: ignore[union-attr]
        for xyxy, conf in zip(xyxy_arr, conf_arr, strict=True):
            detections.append(Detection(xyxy=xyxy, confidence=float(conf)))
        return detections, elapsed
