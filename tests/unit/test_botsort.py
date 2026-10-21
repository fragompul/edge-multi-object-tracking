import numpy as np

from emot.detection.yolo import Detection
from emot.tracking.botsort import BotSortConfig, BotSortTracker
from emot.tracking.bytetrack import ByteTrackConfig, ByteTracker
from emot.tracking.kalman import KalmanBoxTracker


def _scene(n_frames: int = 40, seed: int = 0) -> list[list[Detection]]:
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_frames):
        dets = []
        for k in range(6):
            x, y = 80 + 140 * k + 2.5 * t, 120 + 25 * (k % 3)
            box = np.array([x, y, x + 50, y + 130]) + rng.normal(0, 1.0, 4)
            conf = float(rng.uniform(0.15, 0.95))
            if rng.random() > 0.1:
                dets.append(Detection(xyxy=box, confidence=conf))
        frames.append(dets)
    return frames


def test_botsort_without_cmc_or_reid_is_exactly_bytetrack():
    shared = dict(
        high_conf_threshold=0.5,
        low_conf_threshold=0.1,
        match_iou_threshold=0.7,
        low_match_iou_threshold=0.5,
        min_hits=2,
        max_age=20,
    )
    byte = ByteTracker(ByteTrackConfig(**shared))
    bot = BotSortTracker(BotSortConfig(**shared, use_cmc=False, use_reid=False))
    for dets in _scene():
        a = {tuple(np.round(t.xyxy, 6)) for t in byte.update(dets)}
        b = {tuple(np.round(t.xyxy, 6)) for t in bot.update(dets)}
        assert a == b


def test_cmc_keeps_identity_through_a_fast_camera_pan():
    """Camera pans 60px/frame: without compensation the Kalman prediction
    lags by a full box width and the track is lost; with the true warp it holds."""

    def run(use_cmc: bool) -> set[int]:
        tracker = BotSortTracker(
            BotSortConfig(
                high_conf_threshold=0.5, match_iou_threshold=0.7, min_hits=1, use_cmc=use_cmc
            )
        )
        ids: set[int] = set()
        warp = np.array([[1.0, 0.0, 60.0], [0.0, 1.0, 0.0]])
        for t in range(12):
            box = np.array([100.0 + 60 * t, 200.0, 150.0 + 60 * t, 330.0])
            out = tracker.update([Detection(xyxy=box, confidence=0.9)], warp=warp if t else None)
            ids |= {tr.track_id for tr in out}
        return ids

    assert len(run(use_cmc=True)) == 1
    assert len(run(use_cmc=False)) > 1


def test_apply_affine_translates_and_scales_state():
    kf = KalmanBoxTracker(np.array([100.0, 100.0, 140.0, 200.0]))
    kf.mean[4:6] = [2.0, 0.0]
    warp = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, -5.0]])
    cx, cy, _, h = kf.mean[:4].copy()
    kf.apply_affine(warp)
    assert np.allclose(kf.mean[:2], [2 * cx + 10, 2 * cy - 5])
    assert np.isclose(kf.mean[3], 2 * h)
    assert np.allclose(kf.mean[4:6], [4.0, 0.0])
