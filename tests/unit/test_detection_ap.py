import numpy as np
import pytest

from emot.evaluation.detection_ap import evaluate_detections


def _gt(boxes, ignore=None):
    boxes = np.array(boxes, dtype=float).reshape(-1, 4)
    return boxes, np.array(ignore if ignore is not None else [False] * len(boxes), dtype=bool)


def _det(boxes, scores):
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(scores, dtype=float)


def test_perfect_detections_give_ap_one():
    gt = {1: _gt([[0, 0, 10, 20], [50, 50, 70, 90]])}
    det = {1: _det([[0, 0, 10, 20], [50, 50, 70, 90]], [0.9, 0.8])}
    result = evaluate_detections(det, gt)
    assert result.ap50 == pytest.approx(1.0)
    assert result.ap == pytest.approx(1.0)
    assert result.num_gt == 2


def test_no_detections_give_zero():
    result = evaluate_detections({}, {1: _gt([[0, 0, 10, 20]])})
    assert result.ap50 == 0.0 and result.recall50 == 0.0


def test_false_positive_ranked_first_halves_precision():
    gt = {1: _gt([[0, 0, 10, 20]])}
    det = {1: _det([[200, 200, 220, 240], [0, 0, 10, 20]], [0.9, 0.5])}
    result = evaluate_detections(det, gt)
    assert result.ap50 == pytest.approx(0.5, abs=0.01)


def test_localisation_error_hurts_strict_ap_only():
    gt = {1: _gt([[0, 0, 100, 100]])}
    det = {1: _det([[0, 0, 100, 75]], [0.9])}  # IoU = 0.75
    result = evaluate_detections(det, gt)
    assert result.ap50 == pytest.approx(1.0)
    assert 0.4 < result.ap < 0.7


def test_detection_on_ignore_region_is_not_a_false_positive():
    gt = {1: _gt([[0, 0, 10, 20], [50, 50, 70, 90]], ignore=[False, True])}
    det = {1: _det([[0, 0, 10, 20], [50, 50, 70, 90]], [0.9, 0.8])}
    assert evaluate_detections(det, gt).ap50 == pytest.approx(1.0)
