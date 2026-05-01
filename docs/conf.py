"""Sphinx configuration for the pygenstates documentation."""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "pygenstates"
author = "pygenstates contributors"
copyright = f"{date.today():%Y}, {author}"

try:
    from pygenstates import __version__ as release
except Exception:
    release = "0.1.0"

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} {release} documentation"
