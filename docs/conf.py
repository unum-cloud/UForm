# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Unum · UForm"
copyright = "2023-2026, Unum"
author = "Unum"
with open("../VERSION") as version_file:
    release = version_file.read().strip()
# Reaches `custom.js` as `DOCUMENTATION_OPTIONS.VERSION`.
version = release

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "m2r2",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.jquery",
    "sphinxcontrib.googleanalytics",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "*.md"]
m2r_anonymous_references = True

googleanalytics_id = "341385789"
googleanalytics_enabled = True

# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "../assets/unum.png"
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_baseurl = "https://unum-cloud.github.io/UForm/"
