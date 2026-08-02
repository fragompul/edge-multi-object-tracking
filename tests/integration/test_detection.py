"""Integration test against a real Ultralytics YOLO checkpoint (downloads
`yolo11n.pt` on first run, ~5.6MB) -- exercises the actual detector
backend rather than a fake, the same spirit as this portfolio's
industrial-visual-anomaly-detection backend tests using real model
checkpoints instead of mocks.
"""

from __future__ import annotations

import numpy as np

from emot.detection.yolo import PersonDetector


def test_person_detector_runs_on_a_real_image_without_error():
    # A synthetic image (real MOT17 frames aren't guaranteed to be present
    # in every environment this test runs in, e.g. CI) -- this only checks
    # the detector pipeline doesn't crash and returns a well-formed
    # (possibly empty) result, not detection accuracy.
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

    detector = PersonDetector("yolo11n.pt")
    detections, elapsed = detector.detect(image)

    assert isinstance(detections, list)
    assert elapsed >= 0
    for det in detections:
        assert det.xyxy.shape == (4,)
        assert 0.0 <= det.confidence <= 1.0
