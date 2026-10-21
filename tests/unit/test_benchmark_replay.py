import numpy as np
import pandas as pd

from emot.benchmark.caches import CachedDetections, detection_path, embedding_path, run_key
from emot.benchmark.replay import VARIANTS, replay, score


def _write_scene(tmp_path, n_frames=30):
    """Two pedestrians walking apart; one detector cache + matching GT."""
    rows, gt = [], []
    for f in range(1, n_frames + 1):
        for pid, x0, vx in ((1, 100, 4), (2, 600, -4)):
            x = x0 + vx * f
            rows.append((f, x, 200, x + 50, 330, 0.9))
            gt.append((f, pid, x, 200, 50, 130, 1, 1, 1.0))
    arr = np.array(rows, dtype=np.float32)
    path = detection_path("SEQ", "fake", root=tmp_path)
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        frame_id=arr[:, 0].astype(np.int32),
        x1=arr[:, 1],
        y1=arr[:, 2],
        x2=arr[:, 3],
        y2=arr[:, 4],
        conf=arr[:, 5],
        mean_latency_ms=np.array([1.0]),
    )
    gt_df = pd.DataFrame(
        gt,
        columns=[
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "confidence",
            "class",
            "visibility",
        ],
    )
    return path, gt_df


def test_run_key_keeps_default_layout_and_suffixes_other_sizes():
    assert run_key("yolo11n") == "yolo11n"
    assert run_key("yolo11n", 960) == "yolo11n@960"


def test_cached_detections_index_rows_by_frame(tmp_path):
    path, _ = _write_scene(tmp_path)
    cached = CachedDetections(path)
    assert len(cached) == 60 and len(cached.rows(5)) == 2 and len(cached.rows(999)) == 0
    dets = cached.detections(3)
    assert len(dets) == 2 and dets[0].confidence > 0.89


def test_bytetrack_replay_and_scoring_on_a_clean_scene(tmp_path):
    _, gt_df = _write_scene(tmp_path)
    frames = range(1, 31)
    tracks = replay(VARIANTS["bytetrack"], "SEQ", "fake", frames, {}, cache_root=tmp_path)
    metrics = score(gt_df, tracks, frames)
    assert metrics.num_switches == 0
    assert metrics.mota > 0.9 and metrics.idf1 > 0.9 and metrics.hota > 0.9


def test_botsort_without_reid_matches_bytetrack_on_static_scene(tmp_path):
    _, gt_df = _write_scene(tmp_path)
    frames = range(1, 31)
    warps = np.tile(np.array([[1.0, 0, 0], [0, 1.0, 0]]), (31, 1, 1))
    a = replay(VARIANTS["bytetrack"], "SEQ", "fake", frames, {}, cache_root=tmp_path)
    b = replay(VARIANTS["botsort_cmc"], "SEQ", "fake", frames, {}, cache_root=tmp_path, warps=warps)
    assert score(gt_df, a, frames).hota == score(gt_df, b, frames).hota


def test_embedding_cache_path_layout(tmp_path):
    p = embedding_path("MOT17-02", "yolo11s", "osnet_x0_25", root=tmp_path)
    assert p.name == "yolo11s__osnet_x0_25.npy" and p.parent.name == "MOT17-02"
