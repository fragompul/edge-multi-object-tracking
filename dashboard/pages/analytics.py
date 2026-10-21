"""Analytics over a replayed tracker: trajectories, occupancy heatmap, line crossings and dwell time."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from data_loader import available_sequences, load_sequence_ground_truth, tracks_for
from theme import TEMPLATE, empty

from emot.analytics.events import Zone, dwell_stats
from emot.analytics.trajectories import CountingLine, build_trajectories, occupancy_heatmap
from emot.benchmark.replay import VARIANTS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "mot17"
BACKBONES = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m"]

dash.register_page(__name__, path="/analytics", name="Analytics")


def layout() -> html.Div:
    sequences = available_sequences(DATA_ROOT)
    return html.Div(
        [
            html.H2("Video Analytics"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="an-sequence",
                            clearable=False,
                            value=sequences[0] if sequences else None,
                            options=[{"label": s, "value": s} for s in sequences],
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="an-backbone",
                            clearable=False,
                            value="yolo11s",
                            options=[{"label": b, "value": b} for b in BACKBONES],
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="an-tracker",
                            clearable=False,
                            value="botsort",
                            options=[{"label": v.label, "value": k} for k, v in VARIANTS.items()],
                        ),
                        md=3,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(id="an-counts-summary", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="an-trajectories"), md=6),
                    dbc.Col(dcc.Graph(id="an-heatmap"), md=6),
                ]
            ),
        ]
    )


@callback(
    Output("an-trajectories", "figure"),
    Output("an-heatmap", "figure"),
    Output("an-counts-summary", "children"),
    Input("an-sequence", "value"),
    Input("an-backbone", "value"),
    Input("an-tracker", "value"),
)
def update_analytics(sequence: str | None, backbone: str, tracker: str):
    if sequence is None:
        return empty(""), empty(""), "Select a sequence."
    try:
        raw = tracks_for(sequence, backbone, tracker, DATA_ROOT, REPO_ROOT / "cache")
    except FileNotFoundError:
        return empty(""), empty(""), f"No cached detections for {sequence}/{backbone} yet."
    info, _ = load_sequence_ground_truth(DATA_ROOT, sequence)
    frames = {f: {tid: xyxy for tid, xyxy, _ in rows} for f, rows in raw.items()}
    trajectories = build_trajectories(frames)

    traj = go.Figure()
    for tid, h in trajectories.items():
        if len(h.foot_points) < 2:
            continue
        xs, ys = zip(*h.foot_points, strict=True)
        traj.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines", showlegend=False, line={"width": 1.5}, name=f"track {tid}"
            )
        )
    traj.update_xaxes(visible=False, range=[0, info.im_width])
    traj.update_yaxes(visible=False, range=[info.im_height, 0], scaleanchor="x")
    traj.update_layout(
        template=TEMPLATE,
        title=f"Foot-point trajectories ({len(trajectories)} tracks)",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    heat = occupancy_heatmap(trajectories, info.im_width, info.im_height, bins=40)
    hm = go.Figure(go.Heatmap(z=heat.T, colorscale="Magma", showscale=False))
    hm.update_xaxes(visible=False)
    hm.update_yaxes(visible=False, autorange="reversed")
    hm.update_layout(
        template=TEMPLATE, title="Occupancy heatmap", margin={"l": 0, "r": 0, "t": 40, "b": 0}
    )

    # Defaults a real deployment would configure: a vertical line through the middle and the centre zone.
    from emot.analytics.events import crossing_events

    mid = info.im_width / 2
    events = crossing_events(
        trajectories, CountingLine(mid, 0, mid, info.im_height), info.frame_rate
    )
    a = sum(e.details["direction"] == "a_to_b" for e in events)
    zone = Zone.from_normalized(
        "center",
        [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
        info.im_width,
        info.im_height,
    )
    dwell = dwell_stats(trajectories, zone, info.frame_rate)
    secs = np.array([d["seconds"] for d in dwell.values()]) if dwell else np.zeros(0)
    summary = dbc.Alert(
        f"Middle line: {a} crossings A->B, {len(events) - a} B->A. Centre zone: {len(dwell)} people entered, "
        f"mean stay {secs.mean() if len(secs) else 0:.1f} s, longest {secs.max() if len(secs) else 0:.1f} s. "
        f"({len(trajectories)} anonymous track IDs in total.)",
        color="info",
    )
    return traj, hm, summary
