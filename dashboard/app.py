"""EMOT Lab -- Plotly Dash app (`use_pages`): benchmark explorer, tracker comparison, sequence
playback and analytics, all reading `results/` and `cache/` from disk. The public showcase is the
static web app (`web/`); this lab is the local, Python-powered companion for digging into the caches."""

from __future__ import annotations

import os
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import theme  # noqa: F401  (registers the Plotly template)
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
    title="EMOT Lab",
    assets_folder=str(Path(__file__).parent / "assets"),
)
server = app.server

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                [
                    html.Img(src=app.get_asset_url("favicon.svg"), height="30px", className="me-2"),
                    "EMOT ",
                    html.Small("Lab", className="text-muted"),
                ],
                href="/",
            ),
            dbc.Nav(
                [dbc.NavLink(p["name"], href=p["path"]) for p in dash.page_registry.values()],
                navbar=True,
                className="ms-auto flex-wrap",
            ),
        ],
        fluid=True,
    ),
    dark=True,
    className="mb-4 navbar-expand-lg",
)

app.layout = html.Div([navbar, dbc.Container(dash.page_container, fluid=True, className="pb-5")])

if __name__ == "__main__":
    # Reloader off by default: it forks a child process, which complicates the CI smoke test
    # (needs one killable PID). Opt in with DASH_DEBUG=1.
    app.run(
        debug=os.environ.get("DASH_DEBUG") == "1",
        host=os.environ.get("DASH_HOST", "127.0.0.1"),
        port=8050,
    )
