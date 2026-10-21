"""Builders for the tracker-independent feature caches (ReID embeddings and
camera motion), so that replaying a tracker over a sequence is pure CPU
bookkeeping -- no image decoding, no neural-net forward pass."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from emot.benchmark.caches import (
    CACHE_ROOT,
    DEFAULT_IMGSZ,
    EMBEDDING_MIN_CONF,
    CachedDetections,
    cmc_path,
    detection_path,
    embedding_path,
)
from emot.data.mot17 import frame_path, iter_frame_indices, load_seqinfo
from emot.reid import Embedder, build_embedder
from emot.tracking.cmc import CameraMotionEstimator

logger = logging.getLogger(__name__)


def crop_rgb(image_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Clamped RGB crop; degenerate boxes become a tiny black patch instead of raising."""
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
    x2, y2 = min(w, int(xyxy[2])), min(h, int(xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((4, 4, 3), dtype=np.uint8)
    return cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def build_embeddings(
    sequence: str,
    backbone: str,
    embedders: dict[str, Embedder],
    data_root: Path,
    cache_root: Path = CACHE_ROOT,
    imgsz: int = DEFAULT_IMGSZ,
) -> None:
    """One pass over the frames feeds every requested embedder. Missing-only:
    embedders whose cache file already exists are skipped."""
    todo = {
        name: emb
        for name, emb in embedders.items()
        if not embedding_path(sequence, backbone, name, imgsz, cache_root).exists()
    }
    if not todo:
        return
    cached = CachedDetections(detection_path(sequence, backbone, imgsz, cache_root))
    seq_dir = data_root / sequence
    out = {name: np.zeros((len(cached), emb.dim), dtype=np.float16) for name, emb in todo.items()}
    for frame in iter_frame_indices(load_seqinfo(seq_dir)):
        rows = cached.rows(frame)
        rows = rows[cached.conf[rows] >= EMBEDDING_MIN_CONF]
        if len(rows) == 0:
            continue
        image = cv2.imread(str(frame_path(seq_dir, frame)))
        if image is None:
            raise FileNotFoundError(f"missing frame {frame} in {seq_dir}")
        crops = [crop_rgb(image, cached.boxes[i]) for i in rows]
        for name, emb in todo.items():
            out[name][rows] = emb.embed_crops(crops).astype(np.float16)
    for name, arr in out.items():
        path = embedding_path(sequence, backbone, name, imgsz, cache_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        logger.info("embeddings cached: %s (%s)", path, arr.shape)


def build_cmc(sequence: str, data_root: Path, cache_root: Path = CACHE_ROOT) -> np.ndarray:
    path = cmc_path(sequence, cache_root)
    if path.exists():
        return np.load(path)
    seq_dir = data_root / sequence
    info = load_seqinfo(seq_dir)
    estimator = CameraMotionEstimator()
    warps = []
    for frame in iter_frame_indices(info):
        image = cv2.imread(str(frame_path(seq_dir, frame)))
        if image is None:
            raise FileNotFoundError(f"missing frame {frame} in {seq_dir}")
        warps.append(estimator.estimate(image))
    arr = np.stack(warps)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr


def load_embeddings(
    sequence: str,
    backbone: str,
    embedder: str,
    imgsz: int = DEFAULT_IMGSZ,
    cache_root: Path = CACHE_ROOT,
) -> np.ndarray:
    return np.load(embedding_path(sequence, backbone, embedder, imgsz, cache_root)).astype(
        np.float32
    )


__all__ = ["build_cmc", "build_embeddings", "build_embedder", "crop_rgb", "load_embeddings"]
