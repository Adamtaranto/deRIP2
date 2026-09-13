"""
Shared HTML templating for the deRIP2 reports.

The two report writers (:mod:`derip2.report` and
:mod:`derip2.persequence_report`) build a single self-contained HTML file each.
This package holds what they share: the page shell, the colour theme and its
CSS/JS assets, inline-SVG embedding of matplotlib figures, and the statistics
table cell formatting. Keeping the front-end assets in ``assets/`` (plain
``.css`` / ``.js`` files loaded with :mod:`importlib.resources`) means a restyle
never touches analysis code.
"""

from derip2.reporting.page import render_page
from derip2.reporting.svg import figure_to_svg, inject_svg_tooltips
from derip2.reporting.tables import format_cell, stat_cell
from derip2.reporting.theme import (
    CHROME,
    FIGURE_SURFACE,
    base_style,
    load_asset,
    persequence_script,
    persequence_style,
)

__all__ = [
    'CHROME',
    'FIGURE_SURFACE',
    'base_style',
    'figure_to_svg',
    'format_cell',
    'inject_svg_tooltips',
    'load_asset',
    'persequence_script',
    'persequence_style',
    'render_page',
    'stat_cell',
]
