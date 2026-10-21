import numpy as np
import pytest

from emot.evaluation.hota import HOTAAccumulator


def _box(x: float, y: float = 100.0) -> np.ndarray:
    return np.array([[x, y, 40.0, 80.0]])


def _run(pred_id_for_frame, n_frames: int = 10, pred_present=lambda t: True):
    acc = HOTAAccumulator()
    for t in range(n_frames):
        gt_box = _box(100 + 3 * t)
        if pred_present(t):
            acc.update([1], gt_box, [pred_id_for_frame(t)], gt_box)
        else:
            acc.update([1], gt_box, [], np.zeros((0, 4)))
    return acc.compute()


def test_perfect_tracking_scores_one():
    scores = _run(lambda t: 7)
    assert scores.hota == pytest.approx(1.0)
    assert scores.deta == pytest.approx(1.0)
    assert scores.assa == pytest.approx(1.0)
    assert scores.loca == pytest.approx(1.0)


def test_no_predictions_scores_zero():
    scores = _run(lambda t: 7, pred_present=lambda t: False)
    assert scores.hota == 0.0
    assert scores.deta == 0.0


def test_id_switch_hurts_association_but_not_detection():
    scores = _run(lambda t: 7 if t < 5 else 8)
    assert scores.deta == pytest.approx(1.0)
    # one identity split in two halves: each half covers 5 of the 10 gt frames
    assert scores.assa == pytest.approx(0.5)
    assert scores.hota == pytest.approx(np.sqrt(0.5))


def test_missed_frames_hurt_detection_but_not_association():
    scores = _run(lambda t: 7, pred_present=lambda t: t < 5)
    assert scores.deta == pytest.approx(0.5)
    assert scores.det_pr == pytest.approx(1.0)
    assert scores.det_re == pytest.approx(0.5)
    assert scores.assa == pytest.approx(0.5)


def test_localisation_noise_lowers_loca_and_hota_thresholds():
    acc = HOTAAccumulator()
    for _ in range(10):
        gt_box = _box(100)
        acc.update([1], gt_box, [3], gt_box + np.array([[8, 0, 0, 0]]))
    scores = acc.compute()
    assert 0.5 < scores.loca < 1.0
    assert 0.5 < scores.hota < 1.0
