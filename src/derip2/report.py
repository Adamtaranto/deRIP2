"""
Self-contained HTML report of a deRIP2 strand-bias analysis.

Figures are embedded as inline SVG rather than linked or base64-encoded raster
images: the report stays a single file, the figures remain vector (so they can
be zoomed or lifted straight into a manuscript), and no external asset is ever
fetched.
"""

from html import escape
import logging

from derip2.plotting.strandbias import MODE_TITLES
from derip2.reporting import (
    base_style,
    figure_to_svg,
    format_cell,
    render_page,
)

logger = logging.getLogger(__name__)

# One-paragraph reading guide per strand-bias mode, in reading order: what RIP
# did, what it didn't do, and everything. Headings come from
# :data:`derip2.plotting.strandbias.MODE_TITLES` so the two never drift.
PANEL_BLURBS = {
    'rip': (
        'Columns where an aligned, unmutated substrate dinucleotide shows that '
        'the TpA products arose by RIP. Bars above the axis are forward-strand '
        'events (CA to TA); bars below are reverse-strand events (TG to TA).'
    ),
    'non_rip': (
        'C to T and G to A transitions outside RIP dinucleotide context. A '
        'strand bias here suggests a deamination process other than RIP.'
    ),
    'all_deamination': (
        'Every C and T (forward) and every G and A (reverse), regardless of '
        'context. The backdrop against which the RIP-specific panels should be '
        'read.'
    ),
}
PANELS = tuple((mode, MODE_TITLES[mode], PANEL_BLURBS[mode]) for mode in PANEL_BLURBS)

# Backwards-compatible aliases: the per-sequence report and the test-suite
# import these private names from here.
_STYLE = base_style()
_figure_to_svg = figure_to_svg
_format_cell = format_cell


def _stats_table_html(df):
    """
    Render the statistics DataFrame as an accessible HTML table.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :meth:`derip2.derip.DeRIP.summarize_stats`.

    Returns
    -------
    str
        A ``<table>`` element.
    """
    head = ''.join(f'<th scope="col">{escape(c)}</th>' for c in df.columns)

    rows = []
    for record in df.to_dict('records'):
        cells = []
        for column in df.columns:
            text, css = format_cell(column, record[column])
            cls = f' class="{css}"' if css else ''
            cells.append(f'<td{cls}>{text}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')

    return (
        '<table><thead><tr>'
        + head
        + '</tr></thead><tbody>'
        + ''.join(rows)
        + '</tbody></table>'
    )


def write_html_report(derip, output_file, title=None, ambiguous='split', **kwargs):
    """
    Write a single-file HTML report of the strand-bias analysis.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        A DeRIP object on which ``calculate_rip()`` has already been run.
    output_file : str
        Destination path.
    title : str, optional
        Report heading. Defaults to ``'deRIP2 strand bias report'``.
    ambiguous : {'split', 'exclude', 'weight', 'both'}, optional
        Ambiguity policy used for RSI (default: ``'split'``).
    **kwargs
        Forwarded to each figure, e.g. ``scale``, ``xaxis``, ``columns``.

    Returns
    -------
    str
        The path written.

    Notes
    -----
    Panels that cannot be drawn — for instance because a ``max_columns`` limit
    was passed and exceeded — are reported inline as a note rather than aborting
    the report.
    """
    import matplotlib.pyplot as plt

    df = derip.summarize_stats(ambiguous=ambiguous)
    pooled = derip.rsi_result.pooled()

    panels = []
    for mode, heading, blurb in PANELS:
        try:
            fig = derip.plot_strand_bias(mode=mode, **kwargs)
        except ValueError as exc:
            body = f'<p class="note">Not drawn: {escape(str(exc))}</p>'
        else:
            body = f'<div class="figure">{figure_to_svg(fig, f"{mode}-")}</div>'
            plt.close(fig)
        panels.append(
            f'<section><h2>{escape(heading)}</h2><p>{escape(blurb)}</p>{body}</section>'
        )

    rsi = pooled['RSI']
    if rsi != rsi:  # NaN
        verdict = 'RSI is undefined for this alignment: one strand carries no evidence.'
    elif abs(rsi) < 0.05:
        verdict = (
            'The strands are balanced. Either RIP has not acted, or it has acted '
            'to completion on both strands — compare p_fwd and p_rev to tell '
            'the two apart.'
        )
    else:
        strand = 'forward' if rsi > 0 else 'reverse'
        verdict = f'RIP acted predominantly on the {strand} strand.'

    summary = (
        f'<section><h2>Alignment summary</h2>'
        f'<p>Pooled across all sequences: '
        f'<code>p_fwd = {pooled["p_fwd"]:.3f}</code>, '
        f'<code>p_rev = {pooled["p_rev"]:.3f}</code>, '
        f'<code>RSI = {rsi:+.3f}</code> '
        f'(p = {pooled["pvalue"]:.3g}). '
        f'{escape(verdict)} '
        f'{pooled["n_ambiguous"]} TpA dinucleotides could be attributed to '
        f'either strand and were resolved with the '
        f'<code>{escape(ambiguous)}</code> policy.</p></section>'
    )

    table = (
        '<section><h2>Per-sequence statistics</h2>'
        '<p>RSI is the difference between the proportion of forward and reverse '
        'substrate converted to product. A dash means the value is undefined '
        'because that strand carries neither substrate nor product.</p>'
        f'<div class="table-wrap">{_stats_table_html(df)}</div></section>'
    )

    html = render_page(
        title or 'deRIP2 strand bias report',
        f'{len(derip.alignment)} sequences &times; '
        f'{derip.alignment.get_alignment_length()} columns',
        summary + ''.join(panels) + table,
        css=_STYLE,
    )

    with open(output_file, 'w', encoding='utf-8') as handle:
        handle.write(html)

    logger.info(f'HTML report written to {output_file}')
    return output_file
