from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html
from data_loader import load_detection_summary, load_tracking_results

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "cache" / "detections"

dash.register_page(__name__, path="/", name="Overview")


def _stat_card(label: str, value: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H2(value, className="card-title"),
                html.P(label, className="card-text text-muted"),
            ]
        ),
        className="text-center shadow-sm",
    )


def layout() -> html.Div:
    results = load_tracking_results(RESULTS_DIR)
    detection_summary = load_detection_summary(CACHE_DIR)

    n_sequences = results["sequence"].nunique() if not results.empty else 0
    n_backbones = results["backbone"].nunique() if not results.empty else 0
    n_runs = len(results)
    n_cached_detection_jobs = len(detection_summary)

    return html.Div(
        [
            html.H1("Edge Multi-Object Tracking"),
            html.P(
                "Multi-object detection and tracking on MOT17: a YOLOv8/YOLO11 backbone "
                "speed-vs-accuracy benchmark, ByteTrack vs DeepSORT comparison, ONNX/INT8 "
                "edge export, and derived video analytics (trajectories, counting, heatmaps).",
                className="lead",
            ),
            dbc.Row(
                [
                    dbc.Col(_stat_card("Sequences benchmarked", str(n_sequences)), width=3),
                    dbc.Col(_stat_card("Backbones compared", str(n_backbones)), width=3),
                    dbc.Col(_stat_card("Tracking runs completed", str(n_runs)), width=3),
                    dbc.Col(
                        _stat_card("Detection caches built", str(n_cached_detection_jobs)), width=3
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Explore", className="card-title"),
                        html.Ul(
                            [
                                html.Li(
                                    "Backbone Benchmark -- speed vs. accuracy across YOLO variants"
                                ),
                                html.Li(
                                    "Tracker Comparison -- MOTA/IDF1/ID-switches, "
                                    "ByteTrack vs DeepSORT"
                                ),
                                html.Li(
                                    "Sequence Explorer -- frame-by-frame annotated playback"
                                ),
                                html.Li(
                                    "Analytics -- trajectories, line-crossing counts, heatmaps"
                                ),
                            ]
                        ),
                    ]
                ),
                className="shadow-sm",
            ),
        ]
    )
