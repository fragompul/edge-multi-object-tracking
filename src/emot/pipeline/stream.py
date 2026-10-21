"""Threaded real-time pipeline: capture -> detect+track -> annotate/write.

Three stages joined by bounded queues so a slow stage applies back-pressure
(file sources) or sheds the oldest frames (live sources) instead of growing
latency without bound -- the behaviour a deployed camera pipeline needs.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from emot.detection.yolo import Detection
from emot.pipeline.sources import FrameSource
from emot.tracking.online import OnlineTracker, TrackedBox
from emot.viz.draw import TrackView, TrailBuffer, draw_hud, draw_tracks

_SENTINEL = object()


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> tuple[list[Detection], float]: ...


@dataclass
class StageStats:
    """Exponential moving average of per-frame stage time, in ms."""

    detect_ms: float = 0.0
    track_ms: float = 0.0
    total_ms: float = 0.0
    frames: int = 0

    def push(self, detect_s: float, track_s: float, total_s: float, alpha: float = 0.1) -> None:
        self.frames += 1
        a = 1.0 if self.frames == 1 else alpha
        self.detect_ms += a * (detect_s * 1000 - self.detect_ms)
        self.track_ms += a * (track_s * 1000 - self.track_ms)
        self.total_ms += a * (total_s * 1000 - self.total_ms)

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else 0.0


@dataclass
class PipelineResult:
    frames: int = 0
    wall_seconds: float = 0.0
    dropped: int = 0
    stats: StageStats = field(default_factory=StageStats)

    @property
    def avg_fps(self) -> float:
        return self.frames / self.wall_seconds if self.wall_seconds > 0 else 0.0


@dataclass
class TrackingPipeline:
    detector: Detector
    tracker: OnlineTracker
    label: str = "EMOT"
    trail_length: int = 40
    annotate: bool = True

    def run(
        self,
        source: FrameSource,
        on_frame: Callable[[int, np.ndarray, list[TrackedBox], StageStats], None] | None = None,
        video_out: Path | None = None,
        jsonl_out: Path | None = None,
        max_frames: int | None = None,
        queue_size: int = 8,
    ) -> PipelineResult:
        info = source.info
        in_q: queue.Queue = queue.Queue(maxsize=queue_size)
        out_q: queue.Queue = queue.Queue(maxsize=queue_size)
        result = PipelineResult()
        stop = threading.Event()

        def reader() -> None:
            for i, frame in enumerate(source):
                if stop.is_set() or (max_frames is not None and i >= max_frames):
                    break
                if info.is_live and in_q.full():
                    try:
                        in_q.get_nowait()  # shed the stalest frame
                        result.dropped += 1
                    except queue.Empty:
                        pass
                in_q.put((i + 1, frame))
            in_q.put(_SENTINEL)

        writer = None
        if video_out is not None:
            video_out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(video_out), cv2.VideoWriter.fourcc(*"mp4v"), info.fps, (info.width, info.height)
            )
        jsonl = jsonl_out.open("w") if jsonl_out is not None else None
        trails = TrailBuffer(self.trail_length)

        def sink() -> None:
            while True:
                item = out_q.get()
                if item is _SENTINEL:
                    return
                idx, frame, boxes, stats = item
                if jsonl is not None:
                    jsonl.write(
                        json.dumps(
                            {
                                "frame": idx,
                                "tracks": [
                                    {
                                        "id": b.track_id,
                                        "xyxy": [round(float(v), 1) for v in b.xyxy],
                                        "conf": round(b.confidence, 3),
                                    }
                                    for b in boxes
                                ],
                            }
                        )
                        + "\n"
                    )
                if self.annotate:
                    views = [TrackView(b.track_id, b.xyxy, b.confidence) for b in boxes]
                    trails.push(views)
                    draw_tracks(frame, views, trails.trails())
                    draw_hud(
                        frame,
                        self.label,
                        [
                            f"{stats.fps:5.1f} FPS   det {stats.detect_ms:4.1f} ms   "
                            f"trk {stats.track_ms:4.1f} ms",
                            f"{len(boxes)} people tracked",
                        ],
                    )
                if writer is not None:
                    writer.write(frame)
                if on_frame is not None:
                    on_frame(idx, frame, boxes, stats)

        threads = [
            threading.Thread(target=reader, daemon=True),
            threading.Thread(target=sink, daemon=True),
        ]
        for t in threads:
            t.start()

        start = time.perf_counter()
        try:
            while True:
                item = in_q.get()
                if item is _SENTINEL:
                    break
                idx, frame = item
                t0 = time.perf_counter()
                detections, det_s = self.detector.detect(frame)
                t1 = time.perf_counter()
                boxes = self.tracker.update(frame, detections)
                t2 = time.perf_counter()
                result.stats.push(det_s, t2 - t1, t2 - t0)
                result.frames += 1
                snapshot = StageStats(
                    result.stats.detect_ms,
                    result.stats.track_ms,
                    result.stats.total_ms,
                    result.stats.frames,
                )
                out_q.put((idx, frame, boxes, snapshot))
        finally:
            stop.set()
            out_q.put(_SENTINEL)
            threads[1].join()
            result.wall_seconds = time.perf_counter() - start
            if writer is not None:
                writer.release()
            if jsonl is not None:
                jsonl.close()
            source.close()
        return result
