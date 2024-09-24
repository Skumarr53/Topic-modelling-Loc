# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# Resolve the absolute path to the project root directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------

project = 'Centralized NLP Package'
copyright = '2024, Santhosh Kumar'
author = 'Santhosh Kumar'
release = '0.1.0'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    # "sphinx-search.extension"
    # "sphinx.ext.autosummary",

]

# autosummary_generate = True
# pip install -e .


# open _build/html/index.html

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": False,
}