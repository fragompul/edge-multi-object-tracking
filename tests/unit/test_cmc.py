import cv2
import numpy as np

from emot.tracking.cmc import CameraMotionEstimator


def _textured(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (270, 480), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_first_frame_is_identity():
    warp = CameraMotionEstimator().estimate(_textured())
    assert np.allclose(warp, [[1, 0, 0], [0, 1, 0]])


def test_estimates_a_pure_pan():
    base = _textured()
    shifted = np.roll(base, shift=(6, -10), axis=(0, 1))  # content moves 10 px left, 6 px down
    est = CameraMotionEstimator(downscale=1)
    est.estimate(base)
    warp = est.estimate(shifted)
    assert abs(warp[0, 2] + 10) < 1.0 and abs(warp[1, 2] - 6) < 1.0
    assert abs(warp[0, 0] - 1) < 0.02


def test_static_scene_gives_near_zero_motion():
    base = _textured(1)
    est = CameraMotionEstimator(downscale=1)
    est.estimate(base)
    warp = est.estimate(base.copy())
    assert np.abs(warp[:, 2]).max() < 0.2
