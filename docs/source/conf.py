# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "XPOT"
copyright = "2024, D. F. Thomas du Toit"
author = "D. F. Thomas du Toit"
release = "1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    #    "sphinxext.opengraph",
    #    "sphinx_copybutton",
]

intersphinx_mapping = dict()
intersphinx_mapping["python"] = ("https://docs.python.org/3", None)
intersphinx_mapping["numpy"] = ("https://numpy.org/doc/stable/", None)
intersphinx_mapping["skopt"] = (
    "https://scikit-optimize.github.io/stable/",
    None,
)
intersphinx_mapping["pandas"] = (
    "https://pandas.pydata.org/pandas-docs/stable/",
    None,
)

nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "../../images/xpot-logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}
