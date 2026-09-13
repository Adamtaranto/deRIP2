"""Statistics-table cell formatting shared by the HTML reports."""

from html import escape
import math


def format_cell(column, value):
    """
    Format one statistics-table cell for HTML.

    Parameters
    ----------
    column : str
        Name of the column the value came from; decides the numeric format.
    value : str or float or int or None
        The cell value. NaN and None both render as an en-dash.

    Returns
    -------
    tuple of str
        ``(text, css_class)``, the escaped cell text and the class to style it
        with (empty when the cell needs no styling).
    """
    if isinstance(value, str):
        return escape(value), ''
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '&ndash;', 'muted'

    if column == 'index':
        return str(int(value)), ''
    if column in ('RIP_fwd', 'RIP_rev', 'non_RIP', 'n_ambiguous'):
        return f'{int(value)}', ''
    if column == 'pvalue':
        return f'{value:.3g}', ''
    if column == 'RSI':
        css = 'pos' if value > 0 else ('neg' if value < 0 else '')
        return f'{value:+.3f}', css
    if column in ('fwd_product', 'fwd_substrate', 'rev_product', 'rev_substrate'):
        return f'{value:g}', ''
    return f'{value:.3f}', ''


def stat_cell(column, value, *, sig_class='sig'):
    """
    Format a per-sequence statistic with the RIP-signal flags applied.

    Extends :func:`format_cell` with the two colour rules shared by the
    per-sequence stat cards and the overview table: a positive-RIP Composite
    RIP Index (CRI > 1) and a significant strand-asymmetry p-value (< 0.05)
    are flagged. The derived ``'RIP_total'`` column is rendered as a plain
    integer.

    Parameters
    ----------
    column : str
        Statistic name (a ``summarize_stats`` column, or ``'RIP_total'``).
    value : float or int or None
        The value; ``None``/NaN render as an en-dash.
    sig_class : str, optional
        Class added for a significant p-value (default ``'sig'``; the overview
        table passes ``'pos'``).

    Returns
    -------
    tuple of str
        ``(text, css_class)``.
    """
    if column == 'RIP_total' and value is not None:
        return str(int(value)), ''
    text, css = format_cell(column, value)
    numeric = isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )
    if column == 'CRI' and numeric and value > 1:
        css = 'pos'
    elif column == 'pvalue' and numeric and value < 0.05:
        css = (css + ' ' + sig_class).strip()
    return text, css
