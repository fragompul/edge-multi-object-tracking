from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from data_loader import load_sequence_table, load_tracking_results
from theme import TEMPLATE, TRACKER_COLORS, empty

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

dash.register_page(__name__, path="/tracker-comparison", name="Tracker Comparison")

METRICS = {"hota": "HOTA", "idf1": "IDF1", "mota": "MOTA", "assa": "AssA", "deta": "DetA"}


def layout() -> html.Div:
    results = load_tracking_results(RESULTS_DIR)
    seqs = load_sequence_table(RESULTS_DIR)
    sequences = sorted(results["sequence"].unique()) if not results.empty else []
    return html.Div(
        [
            html.H2("Tracker Comparison"),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="tc-camera",
                            clearable=False,
                            value="all",
                            options=[
                                {"label": "All scenes", "value": "all"},
                                {"label": "Static camera", "value": "static"},
                                {"label": "Moving camera", "value": "moving"},
                            ],
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="tc-sequence-filter",
                            multi=True,
                            value=sequences,
                            options=[{"label": s, "value": s} for s in sequences],
                        ),
                        md=9,
                    ),
                ],
                className="mb-3",
            ),
            dcc.Graph(id="tc-metric-comparison"),
            dcc.Graph(id="tc-switches"),
            html.H4("Full results", className="mt-4"),
            dash_table.DataTable(  # type: ignore[attr-defined]
                id="tc-results-table",
                columns=[
                    {"name": c, "id": c}
                    for c in [
                        "sequence",
                        "backbone",
                        "label",
                        "hota",
                        "idf1",
                        "mota",
                        "num_switches",
                    ]
                ],
                data=results.round(4).to_dict("records") if not results.empty else [],
                sort_action="native",
                filter_action="native",
                page_size=15,
                style_header={
                    "backgroundColor": "#121936",
                    "color": "#22d3ee",
                    "fontWeight": "700",
                },
                style_cell={
                    "backgroundColor": "#0f1630",
                    "color": "#e6eaf7",
                    "textAlign": "center",
                    "border": "1px solid rgba(255,255,255,0.08)",
                },
            ),
            html.Div(
                id="tc-camera-note",
                className="text-muted mt-2",
                children=(
                    f"{(seqs['camera'] == 'moving').sum()} moving-camera and {(seqs['camera'] == 'static').sum()} "
                    "static-camera sequences (measured by the camera-motion estimator)."
                    if len(seqs)
                    else ""
                ),
            ),
        ]
    )


def _filtered(selected: list[str] | None, camera: str):
    results = load_tracking_results(RESULTS_DIR)
    seqs = load_sequence_table(RESULTS_DIR)
    if results.empty:
        return results
    keep = set(selected or [])
    if camera != "all" and len(seqs):
        keep &= set(seqs[seqs["camera"] == camera]["sequence"])
    return results[results["sequence"].isin(keep)]


@callback(
    Output("tc-metric-comparison", "figure"),
    Output("tc-switches", "figure"),
    Input("tc-sequence-filter", "value"),
    Input("tc-camera", "value"),
)
def update(selected: list[str], camera: str) -> tuple[go.Figure, go.Figure]:
    df = _filtered(selected, camera)
    if df.empty:
        return empty("No data for this selection"), empty("")
    means = df.groupby(["tracker", "label"])[list(METRICS) + ["num_switches"]].mean().reset_index()
    means = means.sort_values("hota", ascending=False)
    fig = go.Figure()
    for col, name in METRICS.items():
        fig.add_trace(go.Bar(x=means["label"], y=means[col], name=name))
    fig.update_layout(
        template=TEMPLATE, barmode="group", title="Mean metrics by tracker", yaxis_title="score"
    )
    sw = go.Figure(
        go.Bar(
            x=means["label"],
            y=means["num_switches"],
            marker_color=[TRACKER_COLORS.get(t, "#999") for t in means["tracker"]],
        )
    )
    sw.update_layout(template=TEMPLATE, title="Identity switches (mean per run, lower is better)")
    return fig, sw
