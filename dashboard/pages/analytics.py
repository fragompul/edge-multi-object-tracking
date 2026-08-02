"""Derived analytics: trajectories, line-crossing counts, and an
occupancy heatmap, computed live (in-callback) from cached detections run
through ByteTrack for the selected sequence/backbone.

Running ByteTrack live here (rather than pre-computing and caching this
page's output) is deliberate: it only replays already-cached detections
through a Kalman-filter-and-Hungarian-matching loop with no neural
network inference involved, which takes well under a second even for a
1000+ frame sequence -- there's no accuracy/latency trade-off being hidden
by not caching it, just less to keep in sync.
"""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from data_loader import available_sequences, load_sequence_ground_truth

from emot.analytics.trajectories import (
    CountingLine,
    build_trajectories,
    count_line_crossings,
    occupancy_heatmap,
)
from emot.data.mot17 import iter_frame_indices
from emot.detection.yolo import Detection
from emot.tracking.bytetrack import ByteTracker
from emot.training.tracking_runner import load_cached_detections

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "mot17"
CACHE_DIR = REPO_ROOT / "cache" / "detections"
BACKBONES = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m"]

dash.register_page(__name__, path="/analytics", name="Analytics")


def layout() -> html.Div:
    sequences = available_sequences(DATA_ROOT)
    default_sequence = sequences[0] if sequences else None

    return html.Div(
        [
            html.H2("Video Analytics"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="an-sequence",
                            options=[{"label": s, "value": s} for s in sequences],
                            value=default_sequence,
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="an-backbone",
                            options=[{"label": b, "value": b} for b in BACKBONES],
                            value="yolo11n",
                        ),
                        width=4,
                    ),
                ],
                className="mb-3",
            ),
            html.Div(id="an-counts-summary", className="mb-3"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="an-trajectories"), width=6),
                    dbc.Col(dcc.Graph(id="an-heatmap"), width=6),
                ]
            ),
        ]
    )


def _compute_tracks_by_frame(sequence: str, backbone: str) -> dict[int, dict[int, np.ndarray]]:
    cache_path = CACHE_DIR / sequence / f"{backbone}.npz"
    if not cache_path.exists():
        return {}
    detections_by_frame: dict[int, list[Detection]] = load_cached_detections(cache_path)
    seq_info, _ = load_sequence_ground_truth(DATA_ROOT, sequence)

    tracker = ByteTracker()
    frames: dict[int, dict[int, np.ndarray]] = {}
    for frame_idx in iter_frame_indices(seq_info):
        tracks = tracker.update(detections_by_frame.get(frame_idx, []))
        frames[frame_idx] = {t.track_id: t.xyxy for t in tracks}
    return frames


@callback(
    Output("an-trajectories", "figure"),
    Output("an-heatmap", "figure"),
    Output("an-counts-summary", "children"),
    Input("an-sequence", "value"),
    Input("an-backbone", "value"),
)
def update_analytics(sequence: str | None, backbone: str):
    empty_fig = go.Figure().update_layout(template="plotly_dark")
    if sequence is None:
        return empty_fig, empty_fig, "Select a sequence."

    frames = _compute_tracks_by_frame(sequence, backbone)
    if not frames:
        msg = f"No cached detections for {sequence}/{backbone} yet."
        return empty_fig, empty_fig, msg

    seq_info, _ = load_sequence_ground_truth(DATA_ROOT, sequence)
    trajectories = build_trajectories(frames)

    traj_fig = go.Figure()
    for track_id, history in trajectories.items():
        if len(history.centers) < 2:
            continue
        xs, ys = zip(*history.centers, strict=True)
        traj_fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", name=f"track {track_id}", showlegend=False)
        )
    traj_fig.update_xaxes(visible=False, range=[0, seq_info.im_width])
    traj_fig.update_yaxes(visible=False, range=[seq_info.im_height, 0], scaleanchor="x")
    traj_fig.update_layout(
        template="plotly_dark",
        title=f"Trajectories ({len(trajectories)} tracks)",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    heatmap = occupancy_heatmap(trajectories, seq_info.im_width, seq_info.im_height, bins=40)
    heatmap_fig = go.Figure(
        data=go.Heatmap(z=heatmap.T, colorscale="Inferno", showscale=False)
    )
    heatmap_fig.update_xaxes(visible=False)
    heatmap_fig.update_yaxes(visible=False, autorange="reversed")
    heatmap_fig.update_layout(
        template="plotly_dark",
        title="Occupancy heatmap",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    # A vertical counting line at the frame's horizontal midpoint -- a
    # reasonable default for "people crossing through the middle of the
    # scene" without per-sequence tuning, which a real deployment would do.
    mid_x = seq_info.im_width / 2
    line = CountingLine(mid_x, 0, mid_x, seq_info.im_height)
    counts = count_line_crossings(trajectories, line)
    summary = dbc.Alert(
        f"Line crossings at x={mid_x:.0f}: left->right = {counts['a_to_b']}, "
        f"right->left = {counts['b_to_a']} (out of {len(trajectories)} total tracks)",
        color="info",
    )

    return traj_fig, heatmap_fig, summary
