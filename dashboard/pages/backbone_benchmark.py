"""Detector backbones: AP50 and downstream HOTA against *measured* CPU latency (edge benchmark)."""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from data_loader import load_detection_results, load_edge_rows, load_tracking_results
from theme import TEMPLATE, empty

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

dash.register_page(__name__, path="/backbone-benchmark", name="Backbone Benchmark")

RUNTIMES = {
    "torch_cpu": "PyTorch CPU",
    "torch_mps": "PyTorch GPU",
    "onnx_fp32": "ONNX CPU",
    "onnx_int8_dynamic": "INT8 dynamic",
    "onnx_int8_static": "INT8 static",
    "onnx_coreml": "CoreML",
}


def _scatter() -> go.Figure:
    det, trk, edge = (
        load_detection_results(RESULTS_DIR),
        load_tracking_results(RESULTS_DIR),
        load_edge_rows(RESULTS_DIR),
    )
    if det.empty or edge.empty:
        return empty("Run the benchmark and edge stages first")
    lat = edge[(edge["group"] == "main") & (edge["variant"] == "onnx_fp32")].set_index("backbone")[
        "mean_ms"
    ]
    ap = det.groupby("backbone")["ap50"].mean()
    hota = trk[trk["tracker"] == "bytetrack"].groupby("backbone")["hota"].mean()
    fig = go.Figure()
    for bb in ap.index:
        fig.add_trace(
            go.Scatter(
                x=[lat[bb]],
                y=[ap[bb]],
                mode="markers+text",
                text=[bb],
                textposition="top center",
                name=bb,
                marker={"size": 10 + 80 * float(hota[bb]), "opacity": 0.85},
                hovertemplate=f"{bb}<br>AP50 %{{y:.3f}}<br>{lat[bb]:.1f} ms<br>ByteTrack HOTA {hota[bb]:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title="Detection AP50 vs CPU latency (bubble = ByteTrack HOTA)",
        xaxis_title="ONNX CPU latency, ms/frame",
        yaxis_title="AP50",
        showlegend=False,
    )
    return fig


def _latency() -> go.Figure:
    edge = load_edge_rows(RESULTS_DIR)
    if edge.empty:
        return empty("No edge benchmark yet -- run `make edge`")
    main = edge[edge["group"] == "main"]
    fig = go.Figure()
    for variant, label in RUNTIMES.items():
        s = main[main["variant"] == variant]
        fig.add_trace(go.Bar(x=s["backbone"], y=s["mean_ms"], name=label))
    fig.update_layout(
        template=TEMPLATE,
        barmode="group",
        title="Measured latency by runtime",
        yaxis_title="ms / frame (lower is better)",
    )
    return fig


def _resolution() -> go.Figure:
    edge = load_edge_rows(RESULTS_DIR)
    if edge.empty:
        return empty("No edge benchmark yet")
    res = edge[edge["group"] == "resolution"]
    fig = go.Figure()
    for bb, g in res.groupby("backbone"):
        g = g.sort_values("long_side")
        fig.add_trace(
            go.Scatter(
                x=g["mean_ms"],
                y=g["ap50"],
                mode="lines+markers+text",
                text=g["long_side"].astype(int),
                textposition="bottom right",
                name=str(bb),
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title="Input resolution: AP50 vs latency",
        xaxis_type="log",
        xaxis_title="CPU ms/frame (log)",
        yaxis_title="AP50",
    )
    return fig


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Backbone Benchmark"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=_scatter()), md=6),
                    dbc.Col(dcc.Graph(figure=_latency()), md=6),
                ]
            ),
            dbc.Row([dbc.Col(dcc.Graph(figure=_resolution()), md=12)]),
        ]
    )
