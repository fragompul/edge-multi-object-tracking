"""Frame-by-frame annotated playback: pick a sequence + backbone, scrub a
frame slider, see the cached detection boxes overlaid on the real frame.

Boxes are drawn from the *cached raw detections* (`cache/detections/`),
not a live re-run of the tracker -- this page is about inspecting what
the detector saw frame-by-frame, which is the right level for spotting
detector failure modes (missed/duplicate/flickering boxes); full track
identity (which box belongs to which persistent ID) is the Analytics
page's job instead, since overlaying both at once would clutter a single
view.
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from data_loader import available_sequences, load_frame_detections

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "mot17"
CACHE_DIR = REPO_ROOT / "cache" / "detections"
BACKBONES = ["yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m"]

dash.register_page(__name__, path="/sequence-explorer", name="Sequence Explorer")


def layout() -> html.Div:
    sequences = available_sequences(DATA_ROOT)
    default_sequence = sequences[0] if sequences else None

    return html.Div(
        [
            html.H2("Sequence Explorer"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="se-sequence",
                            options=[{"label": s, "value": s} for s in sequences],
                            value=default_sequence,
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="se-backbone",
                            options=[{"label": b, "value": b} for b in BACKBONES],
                            value="yolo11n",
                        ),
                        width=4,
                    ),
                ],
                className="mb-3",
            ),
            dcc.Slider(id="se-frame-slider", min=1, max=1, step=1, value=1),
            dcc.Graph(id="se-frame-view", style={"height": "70vh"}),
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
    n_frames = len(list((DATA_ROOT / sequence / "img1").glob("*.jpg")))
    return max(1, n_frames), 1


@callback(
    Output("se-frame-view", "figure"),
    Input("se-sequence", "value"),
    Input("se-backbone", "value"),
    Input("se-frame-slider", "value"),
)
def update_frame_view(sequence: str | None, backbone: str, frame_idx: int) -> go.Figure:
    fig = go.Figure()
    if sequence is None:
        fig.update_layout(template="plotly_dark", title="Select a sequence")
        return fig

    frame_path = DATA_ROOT / sequence / "img1" / f"{frame_idx:06d}.jpg"
    image = cv2.imread(str(frame_path))
    if image is None:
        fig.update_layout(template="plotly_dark", title=f"Frame not found: {frame_path.name}")
        return fig

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    encoded_str = base64.b64encode(encoded).decode()

    fig.add_layout_image(
        source=f"data:image/jpeg;base64,{encoded_str}",
        x=0,
        y=0,
        sizex=width,
        sizey=height,
        xref="x",
        yref="y",
        sizing="stretch",
        layer="below",
    )

    detections = load_frame_detections(CACHE_DIR, sequence, backbone, frame_idx)
    for x1, y1, x2, y2, conf in detections:
        fig.add_shape(
            type="rect",
            x0=x1,
            y0=y1,
            x1=x2,
            y1=y2,
            line={"color": "#00ff9f", "width": 2},
        )
        fig.add_annotation(
            x=x1,
            y=y1,
            text=f"{conf:.2f}",
            showarrow=False,
            font={"color": "#00ff9f", "size": 10},
            yshift=10,
        )

    fig.update_xaxes(visible=False, range=[0, width])
    fig.update_yaxes(visible=False, range=[height, 0], scaleanchor="x")
    fig.update_layout(
        template="plotly_dark",
        title=f"{sequence} / {backbone} -- frame {frame_idx} ({len(detections)} detections)",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig
