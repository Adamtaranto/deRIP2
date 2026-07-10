"""
Self-contained HTML report of a deRIP2 strand-bias analysis.

Figures are embedded as inline SVG rather than linked or base64-encoded raster
images: the report stays a single file, the figures remain vector (so they can
be zoomed or lifted straight into a manuscript), and no external asset is ever
fetched.
"""

from html import escape
import io
import logging
import re

logger = logging.getLogger(__name__)

# The panels, in reading order: what RIP did, what it didn't do, and everything.
PANELS = (
    (
        'rip',
        'RIP-like mutations',
        'Columns where an aligned, unmutated substrate dinucleotide shows that '
        'the TpA products arose by RIP. Bars above the axis are forward-strand '
        'events (CA to TA); bars below are reverse-strand events (TG to TA).',
    ),
    (
        'non_rip',
        'Non-RIP deamination',
        'C to T and G to A transitions outside RIP dinucleotide context. A '
        'strand bias here suggests a deamination process other than RIP.',
    ),
    (
        'all_deamination',
        'All C/G deamination',
        'Every C and T (forward) and every G and A (reverse), regardless of '
        'context. The backdrop against which the RIP-specific panels should be '
        'read.',
    ),
)

_STYLE = """
:root {
  --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b;
  --ink-2: #52514e; --muted: #898781; --rule: #e1e0d9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --page: #0d0d0d; --ink: #ffffff;
    --ink-2: #c3c2b7; --muted: #898781; --rule: #2c2c2a;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.5rem; background: var(--page); color: var(--ink);
  font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 1.6rem; font-weight: 600; margin: 0 0 .3rem; }
h2 { font-size: 1.1rem; font-weight: 600; margin: 0 0 .4rem; }
.sub { color: var(--ink-2); margin: 0 0 2rem; }
section {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 10px; padding: 1.4rem; margin-bottom: 1.6rem;
}
section p { color: var(--ink-2); margin: 0 0 1rem; max-width: 72ch; }
/* Figures keep their own light surface, as a printed figure does: the
   colourblind-safe palette is only validated against it. */
.figure {
  overflow-x: auto; background: #fcfcfb; border-radius: 6px; padding: .5rem;
}
.figure svg { display: block; max-width: 100%; height: auto; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td {
  text-align: right; padding: .45rem .6rem;
  border-bottom: 1px solid var(--rule); font-variant-numeric: tabular-nums;
}
th { color: var(--ink-2); font-weight: 600; white-space: nowrap; }
th:nth-child(-n+2), td:nth-child(-n+2) { text-align: left; }
tbody tr:hover { background: rgba(127,127,127,.07); }
.pos { color: #006300; } .neg { color: #b4292a; }
@media (prefers-color-scheme: dark) {
  .pos { color: #0ca30c; } .neg { color: #e66767; }
}
footer { color: var(--muted); font-size: 12px; margin-top: 2rem; }
code { background: rgba(127,127,127,.12); padding: .1em .35em; border-radius: 3px; }
"""


def _figure_to_svg(fig, prefix):
    """
    Render a matplotlib figure to an inline SVG fragment with namespaced IDs.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to render.
    prefix : str
        Unique string prepended to every element ID and internal reference.

    Returns
    -------
    str
        The ``<svg>`` element, ready to embed directly in an HTML body.

    Notes
    -----
    Matplotlib reuses the same element IDs in every SVG it writes (glyph
    definitions such as ``DejaVuSans-41``, tick group names, and so on). Several
    figures in one HTML document would therefore share IDs, and a browser
    resolves ``href="#id"`` to the *first* match in the document — so later
    figures would silently borrow the first figure's glyphs. Prefixing every ID
    and every internal reference keeps each figure self-referential.
    """
    buffer = io.StringIO()
    fig.savefig(buffer, format='svg', bbox_inches='tight')
    svg = buffer.getvalue()

    # Drop everything before the opening <svg> tag: an XML declaration or a
    # DOCTYPE inside an HTML body is invalid.
    match = re.search(r'<svg', svg)
    if match:
        svg = svg[match.start() :]

    svg = re.sub(r'\bid="([^"]+)"', rf'id="{prefix}\1"', svg)
    # Covers both href="#x" and xlink:href="#x".
    svg = re.sub(r'href="#([^"]+)"', rf'href="#{prefix}\1"', svg)
    # clip-path="url(#x)", filter="url(#x)", and friends.
    svg = re.sub(r'url\(#([^)]+)\)', rf'url(#{prefix}\1)', svg)
    return svg


def _format_cell(column, value):
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
    import math

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
            text, css = _format_cell(column, record[column])
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
            body = f'<div class="figure">{_figure_to_svg(fig, f"{mode}-")}</div>'
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

    heading = escape(title or 'deRIP2 strand bias report')
    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{heading}</title><style>{_STYLE}</style></head><body><main>'
        f'<h1>{heading}</h1>'
        f'<p class="sub">{len(derip.alignment)} sequences &times; '
        f'{derip.alignment.get_alignment_length()} columns</p>'
        + summary
        + ''.join(panels)
        + table
        + '<footer>Generated by deRIP2. Figures are inline SVG and render on '
        'their own light surface so the colourblind-safe palette holds.</footer>'
        '</main></body></html>'
    )

    with open(output_file, 'w', encoding='utf-8') as handle:
        handle.write(html)

    logger.info(f'HTML report written to {output_file}')
    return output_file
