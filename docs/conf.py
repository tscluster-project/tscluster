# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys 
import os 

sys.path.insert(0, os.path.abspath('../'))

project = 'tscluster'
copyright = '2024, tscluster'
author = 'Jolomi Tosanwumi'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_tabs.tabs',
    # 'sphinxcontrib.napoleon',
    'numpydoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    ]

# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_use_param = False
# napoleon_use_ivar = True
# napoleon_include_init_with_doc = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

mathjax_config = {
    'TeX': {'equationNumbers': {'autoNumber': 'AMS'}},
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
