"""On-disk layout of the benchmark's three caches.

Everything expensive and tracker-independent is computed once and reused
by every tracker variant:

    cache/detections/<seq>/<backbone>[@<imgsz>].npz   YOLO boxes + scores
    cache/embeddings/<seq>/<backbone>[@<imgsz>]__<embedder>.npy   ReID features
    cache/cmc/<seq>.npy                                 (T, 2, 3) camera-motion warps

The default input size (640) keeps the bare `<backbone>` file name so the
original layout stays valid; other sizes carry an `@<imgsz>` suffix.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emot.detection.yolo import Detection

DEFAULT_IMGSZ = 640
DETECTION_CONF_FLOOR = 0.1  # lowest score cached; trackers apply their own thresholds
EMBEDDING_MIN_CONF = 0.2  # detections below this get no embedding (never used by a tracker)

CACHE_ROOT = Path("cache")


def run_key(backbone: str, imgsz: int = DEFAULT_IMGSZ) -> str:
    return backbone if imgsz == DEFAULT_IMGSZ else f"{backbone}@{imgsz}"


def detection_path(
    sequence: str, backbone: str, imgsz: int = DEFAULT_IMGSZ, root: Path = CACHE_ROOT
) -> Path:
    return root / "detections" / sequence / f"{run_key(backbone, imgsz)}.npz"


def embedding_path(
    sequence: str, backbone: str, embedder: str, imgsz: int = DEFAULT_IMGSZ, root: Path = CACHE_ROOT
) -> Path:
    return root / "embeddings" / sequence / f"{run_key(backbone, imgsz)}__{embedder}.npy"


def cmc_path(sequence: str, root: Path = CACHE_ROOT) -> Path:
    return root / "cmc" / f"{sequence}.npy"


class CachedDetections:
    """One (sequence, backbone, imgsz) detection cache, indexed by frame.

    Row order of the arrays is the contract shared with the embedding cache:
    `embeddings[i]` describes `boxes[i]`.
    """

    def __init__(self, path: Path):
        data = np.load(path)
        self.frame_id: np.ndarray = data["frame_id"]
        self.boxes = np.stack([data["x1"], data["y1"], data["x2"], data["y2"]], axis=1).astype(
            float
        )
        self.conf: np.ndarray = data["conf"].astype(float)
        self.mean_latency_ms = (
            float(data["mean_latency_ms"][0]) if "mean_latency_ms" in data else 0.0
        )
        self._rows: dict[int, np.ndarray] = {}
        order = np.argsort(self.frame_id, kind="stable")
        sorted_frames = self.frame_id[order]
        for frame in np.unique(sorted_frames):
            lo = np.searchsorted(sorted_frames, frame, side="left")
            hi = np.searchsorted(sorted_frames, frame, side="right")
            self._rows[int(frame)] = order[lo:hi]

    def rows(self, frame: int) -> np.ndarray:
        return self._rows.get(frame, np.zeros(0, dtype=np.int64))

    def detections(self, frame: int) -> list[Detection]:
        return [
            Detection(xyxy=self.boxes[i], confidence=float(self.conf[i])) for i in self.rows(frame)
        ]

    def __len__(self) -> int:
        return len(self.frame_id)
