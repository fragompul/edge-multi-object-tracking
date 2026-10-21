"""EMOT brand palette and per-identity colours."""

from __future__ import annotations

import colorsys

# Brand tokens (hex). Mirrored in web/src/styles/tokens.css and dashboard/assets/emot.css.
NAVY = "#0B1020"
NAVY_2 = "#121936"
INK = "#E6EAF7"
MUTED = "#8B93B0"
CYAN = "#22D3EE"
MAGENTA = "#F472B6"
AMBER = "#FBBF24"
LIME = "#A3E635"
VIOLET = "#A78BFA"
RED = "#F87171"

TRACKER_COLORS = {
    "bytetrack": CYAN,
    "botsort": MAGENTA,
    "botsort_cmc": VIOLET,
    "botsort_reid": "#FB923C",
    "deepsort": AMBER,
    "deepsort_osnet": LIME,
    "deepsort_osnet_cascade": "#34D399",
}


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def track_color_bgr(track_id: int) -> tuple[int, int, int]:
    """Stable, well-separated, bright colour per identity (golden-ratio hue walk)."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def track_color_hex(track_id: int) -> str:
    """Same hue walk as `track_color_bgr`, as `#rrggbb` (Plotly / web)."""
    b, g, r = track_color_bgr(track_id)
    return f"#{r:02x}{g:02x}{b:02x}"
