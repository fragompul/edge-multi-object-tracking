"""Frame-by-frame playback: any tracker, any backbone, over the real frames.

Tracks come from replaying the chosen tracker over the cached detections (a few seconds for a
whole sequence, memoised), so what you see is exactly what the benchmark scored."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from data_loader import (
    available_sequences,
    load_frame_detections,
    load_sequence_ground_truth,
    tracks_for,
)
from theme import TEMPLATE, empty

from emot.benchmark.replay import VARIANTS
from emot.viz.palette import track_color_hex

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "mot17"
CACHE_DIR = REPO_ROOT / "cache" / "detections"
BACKBONES = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m"]

dash.register_page(__name__, path="/sequence-explorer", name="Sequence Explorer")


def layout() -> html.Div:
    sequences = available_sequences(DATA_ROOT)
    return html.Div(
        [
            html.H2("Sequence Explorer"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="se-sequence",
                            clearable=False,
                            value=sequences[0] if sequences else None,
                            options=[{"label": s, "value": s} for s in sequences],
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="se-backbone",
                            clearable=False,
                            value="yolo11s",
                            options=[{"label": b, "value": b} for b in BACKBONES],
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="se-tracker",
                            clearable=False,
                            value="botsort",
                            options=[{"label": v.label, "value": k} for k, v in VARIANTS.items()],
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dcc.Checklist(
                            id="se-layers",
                            value=["tracks"],
                            inline=True,
                            labelStyle={"marginRight": "14px"},
                            options=[
                                {"label": " tracks", "value": "tracks"},
                                {"label": " raw detections", "value": "det"},
                            ],
                        ),
                        md=3,
                    ),
                ],
                className="mb-3",
            ),
            dcc.Slider(
                id="se-frame-slider",
                min=1,
                max=1,
                step=1,
                value=1,
                tooltip={"always_visible": False},
            ),
            dcc.Graph(id="se-frame-view", style={"height": "72vh"}),
        ]
    )


@callback(
    Output("se-frame-slider", "max"),
    Output("se-frame-slider", "value"),
    Input("se-sequence", "value"),
)
def update_slider_range(sequence: str | None) -> tuple[int, int]:
    if sequence is None:
        return 1, 1
    return max(1, len(list((DATA_ROOT / sequence / "img1").glob("*.jpg")))), 1


@callback(
    Output("se-frame-view", "figure"),
    Input("se-sequence", "value"),
    Input("se-backbone", "value"),
    Input("se-tracker", "value"),
    Input("se-frame-slider", "value"),
    Input("se-layers", "value"),
)
def update_frame_view(
    sequence: str | None, backbone: str, tracker: str, frame_idx: int, layers: list[str]
) -> go.Figure:
    if sequence is None:
        return empty("Select a sequence")
    image = cv2.imread(str(DATA_ROOT / sequence / "img1" / f"{frame_idx:06d}.jpg"))
    if image is None:
        return empty(f"Frame {frame_idx} not found")
    height, width = image.shape[:2]
    _, enc = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    fig = go.Figure()
    fig.add_layout_image(
        source=f"data:image/jpeg;base64,{base64.b64encode(enc).decode()}",
        x=0,
        y=0,
        sizex=width,
        sizey=height,
        xref="x",
        yref="y",
        sizing="stretch",
        layer="below",
    )

    n_tracks = 0
    if "det" in (layers or []):
        for x1, y1, x2, y2, _conf in load_frame_detections(CACHE_DIR, sequence, backbone, frame_idx):
            fig.add_shape(
                type="rect",
                x0=x1,
                y0=y1,
                x1=x2,
                y1=y2,
                line={"color": "rgba(230,234,247,0.7)", "width": 1, "dash": "dot"},
            )
    if "tracks" in (layers or []):
        try:
            tracks = tracks_for(sequence, backbone, tracker, DATA_ROOT, REPO_ROOT / "cache").get(
                frame_idx, []
            )
        except FileNotFoundError:
            return empty(f"No cached detections for {sequence}/{backbone}")
        n_tracks = len(tracks)
        for tid, (x1, y1, x2, y2), _ in tracks:
            color = track_color_hex(tid)
            fig.add_shape(
                type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line={"color": color, "width": 2}
            )
            fig.add_annotation(
                x=x1,
                y=y1,
                text=str(tid),
                showarrow=False,
                yshift=8,
                font={"color": color, "size": 11},
            )
    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[height, 0], scaleanchor="x")
    fig.update_layout(
        template=TEMPLATE,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title=f"{sequence} / {backbone} / {VARIANTS[tracker].label} -- frame {frame_idx} ({n_tracks} tracks)",
    )
    _ = load_sequence_ground_truth
    return fig
