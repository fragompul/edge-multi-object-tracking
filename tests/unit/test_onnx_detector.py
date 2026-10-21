import numpy as np

from emot.detection.onnx_detector import decode_person_boxes, letterbox, nms_xyxy


def test_letterbox_centres_a_16_9_frame_in_a_rectangular_canvas():
    frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    canvas, lb = letterbox(frame, (384, 640))
    assert canvas.shape == (384, 640, 3)
    assert abs(lb.scale - 1 / 3) < 1e-9
    assert lb.pad_x == 0 and lb.pad_y == 12
    assert canvas[0, 0, 0] == 114 and canvas[200, 300, 0] == 200  # grey bar, then content


def test_letterbox_roundtrip_recovers_original_coordinates():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, lb = letterbox(frame, (384, 640))
    x_img, y_img = 320.0, 240.0
    x_in, y_in = x_img * lb.scale + lb.pad_x, y_img * lb.scale + lb.pad_y
    assert abs((x_in - lb.pad_x) / lb.scale - x_img) < 1e-9
    assert abs((y_in - lb.pad_y) / lb.scale - y_img) < 1e-9


def test_decode_keeps_only_confident_person_anchors():
    raw = np.zeros((1, 84, 100))
    raw[0, :4, 0] = [100, 100, 20, 40]
    raw[0, 4, 0] = 0.9  # person
    raw[0, :4, 1] = [200, 200, 20, 40]
    raw[0, 5, 1] = 0.99  # a bicycle: not a person
    raw[0, :4, 2] = [300, 300, 20, 40]
    raw[0, 4, 2] = 0.1  # person but below threshold
    boxes, scores = decode_person_boxes(raw, conf_threshold=0.25)
    assert boxes.shape == (1, 4) and np.isclose(scores[0], 0.9)


def test_decode_accepts_transposed_layout():
    raw = np.zeros((1, 100, 84))  # (anchors, 4 + classes): the transposed export layout
    raw[0, 0, :4] = [10, 10, 4, 8]
    raw[0, 0, 4] = 0.8
    boxes, scores = decode_person_boxes(raw, 0.25)
    assert len(scores) == 1


def test_nms_suppresses_overlapping_lower_score_box():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 400, 400]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7])
    keep = nms_xyxy(boxes, scores, 0.7)
    assert sorted(keep.tolist()) == [0, 2]
    assert nms_xyxy(np.zeros((0, 4)), np.zeros(0), 0.7).size == 0
