# Configuration file for the Sphinx documentation builder.

import importlib.metadata
import os
import sys
from datetime import date

version = importlib.metadata.version("TRSFX")
if not version:
    version = None
sys.path.insert(0, os.path.abspath("../"))

project = "TRSFX"
copyright = f"{date.today().year}, Ryan O'Dea"
author = "Ryan O'Dea"
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
}
templates_path = ["_templates"]
exclude_patterns = []

html_theme = "piccolo_theme"
html_static_path = ["_static"]
