"""EMOT brand for the Dash lab: one Plotly template + the tracker colour map (mirrors
`emot.viz.palette` and the web app's design tokens)."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from emot.viz import palette as P

TRACKER_COLORS = P.TRACKER_COLORS
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"

pio.templates["emot"] = go.layout.Template(
    layout={
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(255,255,255,0.03)",
        "font": {"family": FONT, "color": P.INK},
        "colorway": [P.CYAN, P.MAGENTA, P.AMBER, P.LIME, P.VIOLET, "#FB923C"],
        "xaxis": {"gridcolor": "rgba(255,255,255,0.08)", "zerolinecolor": "rgba(255,255,255,0.15)"},
        "yaxis": {"gridcolor": "rgba(255,255,255,0.08)", "zerolinecolor": "rgba(255,255,255,0.15)"},
        "title": {"font": {"size": 16}},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
        "margin": {"l": 50, "r": 20, "t": 50, "b": 45},
    }
)
pio.templates.default = "emot"
TEMPLATE = "emot"


def empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template=TEMPLATE, title=title)
    return fig
