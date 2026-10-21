"""The ONNX Runtime detector must reproduce the PyTorch detector (same boxes, same scores)."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import ultralytics

from emot.detection.onnx_detector import OnnxPersonDetector
from emot.detection.yolo import PersonDetector
from emot.export.onnx_export import export_onnx
from emot.tracking.iou import iou_matrix


@pytest.fixture(scope="module")
def bus_image():
    path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    image = cv2.imread(str(path))
    assert image is not None
    return image  # 810x1080, four pedestrians


def test_onnx_detector_matches_pytorch(tmp_path, bus_image):
    h, w = bus_image.shape[:2]
    # static shape = Ultralytics' own rectangular size for this aspect ratio
    shape = (
        (int(np.ceil(640 * h / w / 32) * 32), 640)
        if w >= h
        else (640, int(np.ceil(640 * w / h / 32) * 32))
    )
    onnx = export_onnx("yolo11n.pt", tmp_path, shape)
    ref, _ = PersonDetector("yolo11n.pt", 0.25, "cpu", 640).detect(bus_image)
    got, _ = OnnxPersonDetector(onnx, 0.25).detect(bus_image)
    assert len(got) == len(ref) >= 3
    ious = iou_matrix(np.array([d.xyxy for d in got]), np.array([d.xyxy for d in ref]))
    assert ious.max(axis=1).min() > 0.97
