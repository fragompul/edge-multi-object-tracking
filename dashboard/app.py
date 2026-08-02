"""Plotly Dash app entry point.

Multi-page (`use_pages=True`) rather than one monolithic layout, since the
plan calls for a genuinely explorable dashboard (sequence playback,
backbone benchmark, tracker comparison, analytics) and cramming all four
into one page/callback graph would make the callback dependency chains
hard to reason about. Each page in `dashboard/pages/` owns its own layout
and callbacks; this file only wires up the shell (navbar + page
container) and app-wide config (paths, theme).
"""

from __future__ import annotations

from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "cache" / "detections"
DATA_ROOT = REPO_ROOT / "data" / "mot17"

app = Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    title="Edge Multi-Object Tracking",
    assets_folder=str(Path(__file__).parent / "assets"),
)
server = app.server

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink(page["name"], href=page["path"]))
        for page in dash.page_registry.values()
    ],
    brand="Edge Multi-Object Tracking",
    brand_href="/",
    color="dark",
    dark=True,
    className="mb-4",
)

app.layout = html.Div(
    [
        navbar,
        dbc.Container(dash.page_container, fluid=True, className="pb-5"),
    ]
)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
