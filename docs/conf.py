# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date
import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute
sys.path.insert(0, os.path.abspath('../CHAP/'))

project = 'ChessAnalysisPipeline'
copyright = f'{date.today().year}, CHESSComputing'
author = 'V. Kuznetsov, K. Soloway, R. Verberg'
version = release = 'PACKAGE_VERSION'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_math_dollar',
    'myst_parser',
#    'sphinxcontrib.autodoc_pydantic',
]
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'chap_cli.md',
    'common.md',
    'Galaxy.md',
    'ShedTool.md',
    'CHAP.hdrm.rst',
    'CHAP.inference.rst',
    'CHAP.sin2psi.rst',
    'CHAP.test.rst',
    'CHAP.test.common.rst',
    'workflows/HDRM.md',
]
source_suffix = ['.rst', '.md']
templates_path = ['_templates']

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# Use the myst_enable_extensions option to allow inline ($...$) and display
# ($$...$$) math syntax in Markdown files
myst_enable_extensions = [
    "dollarmath",
    "amsmath", # Optional: for advanced LaTeX environments like 'align'
]
mathjax3_config = {
    'tex2jax': {
        'inlineMath': [ ["\\(","\\)"] ],
        'displayMath': [["\\[","\\]"] ],
    },
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
    }
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'classic'
html_theme = 'sphinx_rtd_theme' # Read the Docs theme
#html_theme = 'sphinxdoc'    # Sphinx ducumentation theme
html_title = 'CHESS Analysis Pipeline'
html_short_title = 'CHAP'
html_static_path = ['_static']
html_css_files = ['center_fig_only.css']
html_show_copyright = False
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html'
    ],
    'using/windows': [
        'windows-sidebar.html',
        'searchbox.html'
    ],
}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

