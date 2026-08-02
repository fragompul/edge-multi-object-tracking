"""Dash's `use_pages` auto-discovery imports each file in this folder as
a bare top-level module (not via normal package-relative imports), so
`dashboard/` itself needs to be on `sys.path` before any page module runs
`from data_loader import ...`. This runs exactly once, the first time
this subpackage is imported -- which Dash's page loader always does
before importing any individual page module.
"""

import sys
from pathlib import Path

_dashboard_dir = str(Path(__file__).resolve().parent.parent)
if _dashboard_dir not in sys.path:
    sys.path.insert(0, _dashboard_dir)
