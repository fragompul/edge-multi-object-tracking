"""Frame sources: video file, webcam index, network stream, or an image folder
(MOTChallenge `img1/` layout works out of the box)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SourceInfo:
    fps: float
    width: int
    height: int
    total_frames: int | None  # None for live sources
    is_live: bool


class FrameSource:
    def __init__(self, spec: str):
        self.spec = spec
        path = Path(spec)
        self._images: list[Path] | None = None
        self._cap: cv2.VideoCapture | None = None
        if path.is_dir():
            folder = path / "img1" if (path / "img1").is_dir() else path
            self._images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            if not self._images:
                raise FileNotFoundError(f"no images found in {folder}")
            first = cv2.imread(str(self._images[0]))
            if first is None:
                raise OSError(f"cannot read {self._images[0]}")
            fps = 30.0
            ini = path / "seqinfo.ini"
            if ini.exists():
                for line in ini.read_text().splitlines():
                    if line.lower().startswith("framerate="):
                        fps = float(line.split("=")[1])
            self.info = SourceInfo(fps, first.shape[1], first.shape[0], len(self._images), False)
        else:
            is_index = spec.isdigit()
            is_live = is_index or "://" in spec
            if not is_index and not is_live and not path.exists():
                raise FileNotFoundError(f"video not found: {spec}")
            cap = cv2.VideoCapture(int(spec) if is_index else spec)
            if not cap.isOpened():
                raise OSError(f"cannot open video source {spec!r}")
            self._cap = cap
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.info = SourceInfo(
                cap.get(cv2.CAP_PROP_FPS) or 30.0,
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                total if total > 0 and not is_live else None,
                is_live,
            )

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._images is not None:
            for p in self._images:
                img = cv2.imread(str(p))
                if img is None:
                    raise OSError(f"cannot read {p}")
                yield img
            return
        assert self._cap is not None
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
