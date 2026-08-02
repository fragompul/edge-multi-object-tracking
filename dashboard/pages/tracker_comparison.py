from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from data_loader import load_tracking_results

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

dash.register_page(__name__, path="/tracker-comparison", name="Tracker Comparison")


def layout() -> html.Div:
    results = load_tracking_results(RESULTS_DIR)
    sequences = sorted(results["sequence"].unique()) if not results.empty else []

    return html.Div(
        [
            html.H2("Tracker Comparison -- ByteTrack vs DeepSORT"),
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        id="tc-sequence-filter",
                        options=[{"label": s, "value": s} for s in sequences],
                        value=sequences,
                        multi=True,
                        placeholder="Filter by sequence...",
                    ),
                    width=6,
                ),
                className="mb-3",
            ),
            dcc.Graph(id="tc-metric-comparison"),
            html.H4("Full results", className="mt-4"),
            dash_table.DataTable(  # type: ignore[attr-defined]  # dash's stubs omit dash_table
                id="tc-results-table",
                columns=[
                    {"name": c, "id": c}
                    for c in [
                        "sequence",
                        "backbone",
                        "tracker",
                        "mota",
                        "idf1",
                        "num_switches",
                        "mostly_tracked",
                        "mostly_lost",
                    ]
                ],
                data=results.to_dict("records") if not results.empty else [],
                sort_action="native",
                filter_action="native",
                page_size=15,
                style_header={"backgroundColor": "#2f3640", "color": "white"},
                style_cell={"backgroundColor": "#1e272e", "color": "white", "textAlign": "center"},
            ),
        ]
    )


@callback(
    Output("tc-metric-comparison", "figure"),
    Input("tc-sequence-filter", "value"),
)
def update_metric_comparison(selected_sequences: list[str]) -> go.Figure:
    results = load_tracking_results(RESULTS_DIR)
    fig = go.Figure()
    if results.empty:
        fig.update_layout(template="plotly_dark", title="No results yet")
        return fig

    filtered = results[results["sequence"].isin(selected_sequences or [])]
    if filtered.empty:
        fig.update_layout(template="plotly_dark", title="No data for the selected sequences")
        return fig

    grouped = filtered.groupby("tracker")[["mota", "idf1"]].mean().reset_index()
    fig.add_trace(go.Bar(x=grouped["tracker"], y=grouped["mota"], name="Mean MOTA"))
    fig.add_trace(go.Bar(x=grouped["tracker"], y=grouped["idf1"], name="Mean IDF1"))
    fig.update_layout(
        template="plotly_dark",
        title="Mean MOTA / IDF1 by tracker (selected sequences)",
        barmode="group",
        yaxis_title="Score",
    )
    return fig
