from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html
from data_loader import load_edge_rows, load_tracking_results

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

dash.register_page(__name__, path="/", name="Overview")


def _stat(label: str, value: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [html.Div(value, className="stat-value"), html.Div(label, className="text-muted")]
        ),
        className="text-center",
    )


def layout() -> html.Div:
    trk = load_tracking_results(RESULTS_DIR)
    edge = load_edge_rows(RESULTS_DIR)
    best = (
        trk.groupby("label")["hota"].mean().sort_values(ascending=False) if not trk.empty else None
    )
    return html.Div(
        [
            html.H1(["See every identity. ", html.Span("On the edge.", className="grad")]),
            html.P(
                "The local lab: explore the benchmark (7 MOT17 sequences x 6 YOLO backbones x 7 tracker "
                "variants), replay any tracker over the cached detections, and derive analytics. "
                "The public showcase is the static EMOT Studio web app.",
                className="lead text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(_stat("tracking runs", str(len(trk))), md=3),
                    dbc.Col(
                        _stat("sequences", str(trk["sequence"].nunique() if len(trk) else 0)), md=3
                    ),
                    dbc.Col(_stat("edge measurements", str(len(edge))), md=3),
                    dbc.Col(
                        _stat(
                            "best mean HOTA", f"{best.iloc[0]:.3f}" if best is not None else "--"
                        ),
                        md=3,
                    ),
                ],
                className="g-3 mb-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Pages"),
                        html.Ul(
                            [
                                html.Li(
                                    "Backbone Benchmark -- detector AP, tracking quality and measured edge latency"
                                ),
                                html.Li(
                                    "Tracker Comparison -- HOTA / IDF1 / MOTA / switches for every variant, by scene"
                                ),
                                html.Li(
                                    "Sequence Explorer -- replay a tracker frame by frame over the real video"
                                ),
                                html.Li(
                                    "Analytics -- trajectories, heatmap, line crossings, dwell in a zone"
                                ),
                            ]
                        ),
                    ]
                )
            ),
        ]
    )
