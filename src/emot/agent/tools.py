"""The analytics tool layer the agent calls.

Design rule: **the model never counts.** Every number in an answer comes from
a deterministic function over the tracks; the LLM only chooses which function
to call and phrases the result. That keeps answers auditable (each carries
the exact tool call that produced it) and immune to arithmetic hallucination.

Coordinates in tool arguments are **normalised** (0-1, origin top-left) so the
model never needs the video resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emot.analytics.events import (
    Zone,
    crossing_events,
    detect_crowding,
    dwell_stats,
    motion_stats,
    occupancy_series,
    scene_summary,
)
from emot.analytics.trajectories import CountingLine, TrackHistory, build_trajectories

REGION_PRESETS: dict[str, list[tuple[float, float]]] = {
    "full": [(0, 0), (1, 0), (1, 1), (0, 1)],
    "left": [(0, 0), (0.5, 0), (0.5, 1), (0, 1)],
    "right": [(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
    "top": [(0, 0), (1, 0), (1, 0.5), (0, 0.5)],
    "bottom": [(0, 0.5), (1, 0.5), (1, 1), (0, 1)],
    "center": [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
}


@dataclass
class TrackStore:
    """Tracks of one video, indexed for the analytics tools."""

    frames: dict[int, dict[int, np.ndarray]]  # frame -> {track_id: xyxy}
    fps: float
    width: int
    height: int

    def __post_init__(self) -> None:
        self.trajectories: dict[int, TrackHistory] = build_trajectories(self.frames)

    @classmethod
    def from_jsonl(cls, path: Path, fps: float, width: int, height: int) -> TrackStore:
        frames: dict[int, dict[int, np.ndarray]] = {}
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            frames[int(row["frame"])] = {
                int(t["id"]): np.array(t["xyxy"], dtype=float) for t in row["tracks"]
            }
        return cls(frames, fps, width, height)

    @property
    def duration_s(self) -> float:
        return (max(self.frames) / self.fps) if self.frames else 0.0

    # -- helpers ---------------------------------------------------------
    def _window(
        self, start_s: float | None, end_s: float | None
    ) -> dict[int, dict[int, np.ndarray]]:
        lo = 1 if start_s is None else int(start_s * self.fps) + 1
        hi = max(self.frames, default=0) if end_s is None else int(end_s * self.fps)
        return {f: t for f, t in self.frames.items() if lo <= f <= hi}

    def _zone(self, region: str | list | None) -> Zone | None:
        if region is None:
            return None
        if isinstance(region, str):
            if region not in REGION_PRESETS:
                raise ValueError(f"unknown region {region!r}; use one of {sorted(REGION_PRESETS)}")
            return Zone.from_normalized(region, REGION_PRESETS[region], self.width, self.height)
        pts = [(float(x), float(y)) for x, y in region]
        if len(pts) < 3:
            raise ValueError("a polygon needs at least 3 points")
        return Zone.from_normalized("custom", pts, self.width, self.height)

    # -- tools -----------------------------------------------------------
    def scene_summary(self) -> dict:
        return scene_summary(self.frames, self.trajectories, self.fps)

    def count_people(
        self,
        start_s: float | None = None,
        end_s: float | None = None,
        region: str | list | None = None,
    ) -> dict:
        window = self._window(start_s, end_s)
        zone = self._zone(region)
        ids: set[int] = set()
        for tracks in window.values():
            for tid, box in tracks.items():
                if zone is None or zone.contains(((box[0] + box[2]) / 2, box[3])):
                    ids.add(tid)
        series = occupancy_series(window, zone)
        return {
            "unique_people": len(ids),
            "max_simultaneous": max(series.values(), default=0),
            "mean_simultaneous": round(float(np.mean(list(series.values()))), 2) if series else 0.0,
            "window_s": [start_s or 0.0, end_s if end_s is not None else round(self.duration_s, 2)],
            "region": region if region is not None else "full frame",
        }

    def line_crossings(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        start_s: float | None = None,
        end_s: float | None = None,
    ) -> dict:
        line = CountingLine(x1 * self.width, y1 * self.height, x2 * self.width, y2 * self.height)
        lo = 0.0 if start_s is None else start_s
        hi = float("inf") if end_s is None else end_s
        events = [
            e for e in crossing_events(self.trajectories, line, self.fps) if lo <= e.seconds <= hi
        ]
        a = sum(e.details["direction"] == "a_to_b" for e in events)
        return {
            "a_to_b": a,
            "b_to_a": len(events) - a,
            "total": len(events),
            "note": "A is the side where cross(p2-p1, p-p1) < 0 (screen coordinates, y down).",
            "first_events": [
                {
                    "t_s": round(e.seconds, 2),
                    "track_id": e.track_id,
                    "direction": e.details["direction"],
                }
                for e in events[:5]
            ],
        }

    def dwell_time(self, region: str | list, top_k: int = 5) -> dict:
        zone = self._zone(region)
        assert zone is not None
        stats = dwell_stats(self.trajectories, zone, self.fps)
        ranked = sorted(stats.items(), key=lambda kv: -kv[1]["seconds"])
        seconds = [v["seconds"] for v in stats.values()]
        return {
            "people_who_entered": len(stats),
            "mean_dwell_s": round(float(np.mean(seconds)), 2) if seconds else 0.0,
            "max_dwell_s": round(max(seconds), 2) if seconds else 0.0,
            "longest": [
                {"track_id": tid, "seconds": round(v["seconds"], 2), "entries": int(v["entries"])}
                for tid, v in ranked[:top_k]
            ],
        }

    def crowding(
        self, threshold: int, region: str | list | None = None, min_seconds: float = 1.0
    ) -> dict:
        events = detect_crowding(self.frames, self._zone(region), self.fps, threshold, min_seconds)
        return {
            "episodes": [
                {
                    "start_s": round(e.seconds, 2),
                    "end_s": round(e.details["end_frame"] / self.fps, 2),
                    "peak": e.details["peak"],
                }
                for e in events
            ],
            "count": len(events),
        }

    def busiest_moments(self, top_k: int = 3, region: str | list | None = None) -> dict:
        series = occupancy_series(self.frames, self._zone(region))
        per_second: dict[int, list[int]] = {}
        for f, n in series.items():
            per_second.setdefault(int((f - 1) / self.fps), []).append(n)
        ranked = sorted(per_second.items(), key=lambda kv: -float(np.mean(kv[1])))[:top_k]
        return {
            "moments": [
                {"second": s, "mean_people": round(float(np.mean(v)), 1), "peak": max(v)}
                for s, v in ranked
            ]
        }

    def track_info(self, track_id: int) -> dict:
        hist = self.trajectories.get(int(track_id))
        if hist is None:
            return {"error": f"no track with id {track_id}"}
        stats = motion_stats({int(track_id): hist}, self.fps, min_frames=2).get(int(track_id), {})
        return {
            "track_id": int(track_id),
            "first_seen_s": round((hist.frame_indices[0] - 1) / self.fps, 2),
            "last_seen_s": round((hist.frame_indices[-1] - 1) / self.fps, 2),
            "frames_visible": len(hist.frame_indices),
            "start_xy_norm": [
                round(hist.foot_points[0][0] / self.width, 3),
                round(hist.foot_points[0][1] / self.height, 3),
            ],
            "end_xy_norm": [
                round(hist.foot_points[-1][0] / self.width, 3),
                round(hist.foot_points[-1][1] / self.height, 3),
            ],
            **{k: round(v, 2) for k, v in stats.items()},
        }

    # -- dispatch --------------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any]) -> dict:
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name!r}"}
        try:
            return getattr(self, name)(**arguments)
        except (TypeError, ValueError) as exc:
            return {"error": str(exc)}


_REGION = {
    "description": "A preset name (full, left, right, top, bottom, center) or a polygon as a "
    "list of [x, y] normalised points (0-1, origin top-left).",
    "anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    ],
}
_WINDOW = {
    "start_s": {"type": "number", "description": "Window start in seconds (default: video start)."},
    "end_s": {"type": "number", "description": "Window end in seconds (default: video end)."},
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": properties, "required": required},
    }


TOOL_SCHEMAS: list[dict] = [
    _tool(
        "scene_summary",
        "Overall statistics of the video: duration, number of distinct "
        "tracked people, mean/peak occupancy and when the peak happened.",
        {},
        [],
    ),
    _tool(
        "count_people",
        "Count distinct people (unique track IDs) in a time window and/or "
        "region, plus the maximum and mean number visible at once.",
        {**_WINDOW, "region": _REGION},
        [],
    ),
    _tool(
        "line_crossings",
        "Count people crossing a virtual line, per direction. Coordinates are "
        "normalised. Example: a vertical line down the middle is x1=0.5,y1=0,x2=0.5,y2=1.",
        {
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "x2": {"type": "number"},
            "y2": {"type": "number"},
            **_WINDOW,
        },
        ["x1", "y1", "x2", "y2"],
    ),
    _tool(
        "dwell_time",
        "How long people stayed inside a region (seconds per person, standing "
        "point = feet), with the longest stays.",
        {"region": _REGION, "top_k": {"type": "integer"}},
        ["region"],
    ),
    _tool(
        "crowding",
        "Find episodes where at least `threshold` people were in the region at once "
        "for at least `min_seconds`.",
        {"threshold": {"type": "integer"}, "region": _REGION, "min_seconds": {"type": "number"}},
        ["threshold"],
    ),
    _tool(
        "busiest_moments",
        "The seconds of the video with the highest average occupancy.",
        {"top_k": {"type": "integer"}, "region": _REGION},
        [],
    ),
    _tool(
        "track_info",
        "Details of one tracked person: when they appear, where they start and "
        "end, and how fast they move (body-heights per second).",
        {"track_id": {"type": "integer"}},
        ["track_id"],
    ),
]
TOOL_NAMES = {t["name"] for t in TOOL_SCHEMAS}
