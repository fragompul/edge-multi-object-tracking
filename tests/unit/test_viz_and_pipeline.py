import json

import cv2
import numpy as np

from emot.detection.yolo import Detection
from emot.pipeline.sources import FrameSource
from emot.pipeline.stream import TrackingPipeline
from emot.tracking.online import OnlineTracker
from emot.viz.draw import TrackView, TrailBuffer, draw_hud, draw_tracks
from emot.viz.palette import hex_to_bgr, track_color_bgr


def _view(tid, x):
    return TrackView(tid, np.array([x, 50.0, x + 40, 150.0]))


def test_draw_tracks_marks_pixels_and_returns_same_frame():
    frame = np.zeros((200, 300, 3), np.uint8)
    out = draw_tracks(frame, [_view(1, 20), _view(2, 120)])
    assert out is frame and frame.any()


def test_track_colours_are_stable_and_distinct():
    assert track_color_bgr(7) == track_color_bgr(7)
    assert len({track_color_bgr(i) for i in range(1, 30)}) == 29
    assert hex_to_bgr("#22D3EE") == (238, 211, 34)


def test_trails_expire_after_patience():
    buf = TrailBuffer(length=5, patience=2)
    buf.push([_view(1, 10)])
    for _ in range(3):
        buf.push([])
    assert 1 not in buf.trails()
    buf.push([_view(2, 10)])
    assert 2 in buf.trails()


def test_hud_draws_without_error_on_tiny_frames():
    draw_hud(np.zeros((40, 60, 3), np.uint8), "EMOT", ["12 FPS"])


class _FakeDetector:
    """One person walking right, found every frame."""

    def __init__(self):
        self.i = 0

    def detect(self, image):
        self.i += 1
        x = 20 + 4 * self.i
        return [Detection(np.array([x, 40.0, x + 30, 120.0]), 0.9)], 0.001


def test_pipeline_runs_over_an_image_folder_and_writes_jsonl(tmp_path):
    folder = tmp_path / "seq" / "img1"
    folder.mkdir(parents=True)
    for i in range(1, 13):
        cv2.imwrite(str(folder / f"{i:06d}.jpg"), np.full((160, 240, 3), 90, np.uint8))
    (tmp_path / "seq" / "seqinfo.ini").write_text("[Sequence]\nframeRate=25\n")
    src = FrameSource(str(tmp_path / "seq"))
    assert src.info.fps == 25 and src.info.total_frames == 12 and not src.info.is_live

    out_jsonl = tmp_path / "tracks.jsonl"
    pipe = TrackingPipeline(_FakeDetector(), OnlineTracker("bytetrack", tuned={}), label="test")
    result = pipe.run(src, jsonl_out=out_jsonl, video_out=tmp_path / "o.mp4")
    assert result.frames == 12 and result.avg_fps > 0
    rows = [json.loads(line) for line in out_jsonl.read_text().splitlines()]
    assert len(rows) == 12
    ids = {t["id"] for r in rows for t in r["tracks"]}
    assert len(ids) == 1  # one identity kept across the clip
    assert (tmp_path / "o.mp4").stat().st_size > 0
