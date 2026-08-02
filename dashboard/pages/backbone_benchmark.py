"""Speed-vs-accuracy backbone comparison.

"Accuracy" here is each backbone's mean tracking MOTA (averaged across
sequences, ByteTrack only, to hold the tracker fixed while varying the
backbone) rather than a separate detection-only mAP: this project doesn't
compute per-backbone detection mAP against MOT17's box-level ground truth
independently of tracking (that would need its own IoU-matching/mAP
machinery on top of what tracking evaluation already provides), so
tracking MOTA is used as the accuracy signal instead -- documented in
docs/limitations.md as a deliberate scope trade-off, and a defensible one
since tracking quality is the metric that actually matters end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from data_loader import load_detection_summary, load_export_benchmark, load_tracking_results

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "cache" / "detections"

dash.register_page(__name__, path="/backbone-benchmark", name="Backbone Benchmark")

BACKBONE_COLORS = {
    "yolov8n": "#7f8fa6",
    "yolov8s": "#535c68",
    "yolov8m": "#2f3640",
    "yolo11n": "#00a8ff",
    "yolo11s": "#0097e6",
    "yolo11m": "#0652DD",
}


def _speed_accuracy_figure() -> go.Figure:
    results = load_tracking_results(RESULTS_DIR)
    detection_summary = load_detection_summary(CACHE_DIR)

    fig = go.Figure()
    if results.empty or not detection_summary:
        fig.update_layout(
            template="plotly_dark", title="No results yet -- run the benchmark pipeline first"
        )
        return fig

    bytetrack = results[results["tracker"] == "bytetrack"]
    mean_mota_by_backbone = bytetrack.groupby("backbone")["mota"].mean()

    latency_by_backbone: dict[str, list[float]] = {}
    for key, entry in detection_summary.items():
        _, backbone = key.split("/")
        latency_by_backbone.setdefault(backbone, []).append(entry["mean_latency_ms"])

    for backbone, latencies in latency_by_backbone.items():
        if backbone not in mean_mota_by_backbone.index:
            continue
        mean_latency = sum(latencies) / len(latencies)
        fig.add_trace(
            go.Scatter(
                x=[mean_latency],
                y=[mean_mota_by_backbone[backbone]],
                mode="markers+text",
                name=backbone,
                text=[backbone],
                textposition="top center",
                marker={"size": 16, "color": BACKBONE_COLORS.get(backbone, "#e84118")},
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title="Detection speed vs. tracking accuracy (mean MOTA, ByteTrack)",
        xaxis_title="Mean detection latency (ms/frame, CPU)",
        yaxis_title="Mean MOTA across sequences",
        showlegend=False,
    )
    return fig


def _export_benchmark_figure() -> go.Figure:
    export_df = load_export_benchmark(RESULTS_DIR)
    fig = go.Figure()
    if export_df.empty:
        fig.update_layout(
            template="plotly_dark", title="No export benchmark yet -- run export/benchmark.py"
        )
        return fig

    for variant in export_df["variant"].unique():
        subset = export_df[export_df["variant"] == variant]
        fig.add_trace(
            go.Bar(x=subset["backbone"], y=subset["mean_latency_ms"], name=variant)
        )
    fig.update_layout(
        template="plotly_dark",
        title="Edge export latency: PyTorch FP32 vs ONNX FP32 vs ONNX INT8",
        xaxis_title="Backbone",
        yaxis_title="Mean latency (ms/frame, CPU)",
        barmode="group",
    )
    return fig


def layout() -> html.Div:
    return html.Div(
        [
            html.H2("Backbone Benchmark"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=_speed_accuracy_figure()), width=6),
                    dbc.Col(dcc.Graph(figure=_export_benchmark_figure()), width=6),
                ]
            ),
        ]
    )
