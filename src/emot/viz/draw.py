"""Frame annotation: boxes, ID chips, fading trajectory trails, HUD."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from emot.viz.palette import AMBER, CYAN, INK, NAVY, hex_to_bgr, track_color_bgr

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class TrackView:
    """What the renderer needs to know about one tracked box."""

    track_id: int
    xyxy: np.ndarray
    confidence: float = 1.0


class TrailBuffer:
    """Keeps the last `length` foot-points of each identity for fading trails.
    An identity unseen for `patience` frames loses its trail."""

    def __init__(self, length: int = 40, patience: int = 3):
        self.length = length
        self.patience = patience
        self._points: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.length)
        )
        self._unseen: dict[int, int] = {}

    def push(self, tracks: Iterable[TrackView]) -> None:
        seen = set()
        for t in tracks:
            seen.add(t.track_id)
            self._unseen[t.track_id] = 0
            self._points[t.track_id].append((int((t.xyxy[0] + t.xyxy[2]) / 2), int(t.xyxy[3])))
        for tid in [k for k in self._points if k not in seen]:
            self._unseen[tid] = self._unseen.get(tid, 0) + 1
            if self._unseen[tid] > self.patience:
                del self._points[tid]
                del self._unseen[tid]

    def trails(self) -> dict[int, list[tuple[int, int]]]:
        return {k: list(v) for k, v in self._points.items()}


def _alpha_rect(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(img.shape[1], p2[0]), min(img.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_tracks(
    frame: np.ndarray,
    tracks: Sequence[TrackView],
    trails: dict[int, list[tuple[int, int]]] | None = None,
    thickness: int | None = None,
    highlight_ids: set[int] | None = None,
) -> np.ndarray:
    """Draws onto `frame` in place and returns it."""
    h, w = frame.shape[:2]
    th = thickness or max(1, round(min(h, w) / 360))
    font_scale = max(0.4, min(h, w) / 1100)

    if trails:
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            color = track_color_bgr(tid)
            for i in range(1, len(pts)):
                fade = i / len(pts)
                cv2.line(
                    frame, pts[i - 1], pts[i], color, max(1, round(th * 2 * fade)), cv2.LINE_AA
                )

    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t.xyxy)
        color = track_color_bgr(t.track_id)
        _alpha_rect(frame, (x1, y1), (x2, y2), color, 0.10)
        if highlight_ids and t.track_id in highlight_ids:
            cv2.rectangle(
                frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), th + 1, cv2.LINE_AA
            )
        # corner brackets instead of a full rectangle: the brand's "tracking reticle"
        arm = max(6, int(min(x2 - x1, y2 - y1) * 0.22))
        corners = ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1))
        for cx, cy, dx, dy in corners:
            cv2.line(frame, (cx, cy), (cx + dx * arm, cy), color, th + 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * arm), color, th + 1, cv2.LINE_AA)
        label = f"{t.track_id}"
        (tw, tth), base = cv2.getTextSize(label, FONT, font_scale, 1)
        _alpha_rect(frame, (x1, y1 - tth - base - 4), (x1 + tw + 8, y1), color, 0.85)
        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - base - 1),
            FONT,
            font_scale,
            hex_to_bgr(NAVY),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(
    frame: np.ndarray,
    title: str,
    lines: Sequence[str] = (),
    accent: str = CYAN,
) -> np.ndarray:
    """Translucent top-left info panel with an accent bar."""
    h, w = frame.shape[:2]
    scale = max(0.45, min(h, w) / 900)
    pad = int(10 * scale + 4)
    line_h = int(26 * scale)
    texts = [title, *lines]
    widths = [cv2.getTextSize(s, FONT, scale, 1)[0][0] for s in texts]
    box_w = max(widths) + 2 * pad + 6
    box_h = line_h * len(texts) + pad
    _alpha_rect(frame, (pad, pad), (pad + box_w, pad + box_h), hex_to_bgr(NAVY), 0.72)
    cv2.rectangle(frame, (pad, pad), (pad + 4, pad + box_h), hex_to_bgr(accent), -1)
    for i, text in enumerate(texts):
        color = hex_to_bgr(accent if i == 0 else INK)
        cv2.putText(
            frame,
            text,
            (pad + 14, pad + line_h * (i + 1) - int(6 * scale)),
            FONT,
            scale,
            color,
            1 if i else 2,
            cv2.LINE_AA,
        )
    return frame


def draw_counting_line(
    frame: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], counts: dict[str, int]
) -> np.ndarray:
    cv2.line(frame, p1, p2, hex_to_bgr(AMBER), 3, cv2.LINE_AA)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    text = f"A>B {counts.get('a_to_b', 0)}   B>A {counts.get('b_to_a', 0)}"
    scale = max(0.5, frame.shape[0] / 900)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    _alpha_rect(
        frame, (mid[0] - 8, mid[1] - th - 14), (mid[0] + tw + 8, mid[1] + 6), hex_to_bgr(NAVY), 0.8
    )
    cv2.putText(frame, text, (mid[0], mid[1] - 4), FONT, scale, hex_to_bgr(AMBER), 2, cv2.LINE_AA)
    return frame
