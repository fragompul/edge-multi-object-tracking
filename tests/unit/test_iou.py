import numpy as np

from emot.tracking.iou import iou_matrix


def test_identical_boxes_have_iou_one():
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    result = iou_matrix(boxes, boxes)
    assert result.shape == (1, 1)
    assert np.isclose(result[0, 0], 1.0)


def test_disjoint_boxes_have_iou_zero():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[100, 100, 110, 110]], dtype=float)
    result = iou_matrix(a, b)
    assert np.isclose(result[0, 0], 0.0)


def test_half_overlap_known_value():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[5, 0, 15, 10]], dtype=float)
    result = iou_matrix(a, b)
    # intersection = 5x10=50, union = 100+100-50=150
    assert np.isclose(result[0, 0], 50 / 150)


def test_empty_inputs_return_correctly_shaped_array():
    empty = np.zeros((0, 4))
    boxes = np.array([[0, 0, 10, 10]], dtype=float)
    assert iou_matrix(empty, boxes).shape == (0, 1)
    assert iou_matrix(boxes, empty).shape == (1, 0)
    assert iou_matrix(empty, empty).shape == (0, 0)


def test_matrix_shape_for_multiple_boxes():
    a = np.random.default_rng(0).uniform(0, 100, size=(3, 4))
    a[:, 2:] += a[:, :2] + 1  # ensure right > left, bottom > top
    b = np.random.default_rng(1).uniform(0, 100, size=(5, 4))
    b[:, 2:] += b[:, :2] + 1
    result = iou_matrix(a, b)
    assert result.shape == (3, 5)
    assert np.all(result >= 0) and np.all(result <= 1)
