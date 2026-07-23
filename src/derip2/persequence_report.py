"""
Self-contained, interactive per-sequence HTML report.

Where :mod:`derip2.report` gives one alignment-wide view of strand bias, this
module gives one panel *per input sequence*: the alignment row with its RIP
sites highlighted, a fixed-height per-sequence strand-bias strip, a per-sequence
SBS-96 mutation spectrum measured against the reconstructed ancestor, and that
sequence's summary statistics. The panels are stacked in a single HTML file and
shown one at a time; the reader steps between sequences with the arrow keys or
the prev/next buttons.

As with :mod:`derip2.report`, every figure is embedded as inline SVG so the
report is a single self-contained file with no external assets. Each figure is
given a unique ID prefix (``s{row}{kind}-``) because matplotlib reuses element
IDs across figures and a browser resolves ``href="#id"`` to the first match in
the document — without unique prefixes, later figures would borrow the first
figure's glyphs.
"""

import base64
from html import escape
import json
import logging

from tqdm import tqdm

from derip2.plotting.persequence import (
    NONRIP_COLOR,
    PRODUCT_COLOR,
    SUBSTRATE_COLOR,
)
from derip2.plotting.strandbias import BASE_COLORS
from derip2.report import (
    _STYLE,
    _figure_to_svg,
    _format_cell,
)

logger = logging.getLogger(__name__)

# Grouped, transposed statistics layout: (section title, description, [(column,
# row label), ...]). Each group becomes a small card with a stat/value table.
_STAT_SECTIONS = (
    (
        'RIP events',
        'Counts of RIP-attributable deamination in this sequence, split by the '
        'strand the C→T product was read on, plus deaminations outside RIP '
        'dinucleotide context.',
        (
            ('RIP_total', 'Total RIP events'),
            ('RIP_fwd', 'Forward RIP events'),
            ('RIP_rev', 'Reverse RIP events'),
            ('non_RIP', 'Non-RIP deaminations'),
        ),
    ),
    (
        'Strand bias (RSI)',
        'The RIP Strandedness Imbalance: p_fwd and p_rev are the fraction of each '
        'strand’s substrate converted to product; RSI = p_fwd − p_rev '
        '(positive = forward-biased). The p-value tests strand asymmetry; '
        'ambiguous TpA sites could derive from either strand.',
        (
            ('RSI', 'RSI'),
            ('p_fwd', 'p_fwd (forward)'),
            ('p_rev', 'p_rev (reverse)'),
            ('pvalue', 'p-value'),
            ('fwd_product', 'Forward product'),
            ('fwd_substrate', 'Forward substrate'),
            ('rev_product', 'Reverse product'),
            ('rev_substrate', 'Reverse substrate'),
            ('n_ambiguous', 'Ambiguous TpA'),
        ),
    ),
    (
        'Composite RIP Index (CRI)',
        'The classical CRI and its components: the product index (PI, TpA/ApT) '
        'minus the substrate index (SI, (CpA+TpG)/(ApC+GpT)). A positive CRI is '
        'the hallmark of RIP.',
        (
            ('CRI', 'CRI'),
            ('PI', 'Product index (PI)'),
            ('SI', 'Substrate index (SI)'),
        ),
    ),
    (
        'Composition',
        'Base composition of this sequence.',
        (('GC', 'GC content'),),
    ),
)


def _inject_svg_tooltips(svg, prefix, titles, fasta_keys=None):
    """
    Tag named ``<g>`` groups with ``data-tip`` (and optional ``data-fasta``).

    Matplotlib writes an artist's ``gid`` as ``<g id="gid">``; ``_figure_to_svg``
    then namespaces every id with ``prefix``. Adding a ``data-tip`` attribute to
    that group's opening tag lets the report's own JavaScript show a floating
    tooltip with no delay and pin it on click (a native ``<title>`` would impose
    a ~1 s browser delay and never appear on click). ``aria-label`` mirrors the
    text so assistive technology can still announce it. When ``fasta_keys`` maps a
    ``gid`` to a payload key, a ``data-fasta`` attribute is added too so a click on
    that group opens the FASTA popup (see :data:`_PSR_SCRIPT`).

    Both attributes are written in a single pass per ``gid`` because they share
    one opening tag: a second ``str.replace`` for the same ``gid`` would no longer
    match once the tag had grown its first attribute.

    Parameters
    ----------
    svg : str
        The inline SVG fragment (already id-prefixed).
    prefix : str
        The id prefix applied by :func:`derip2.report._figure_to_svg`.
    titles : dict of str to str
        Maps each artist ``gid`` to its tooltip text.
    fasta_keys : dict of str to str, optional
        Maps a ``gid`` to a FASTA-payload key; those groups gain a ``data-fasta``
        attribute. ``gid``\\s present here but absent from ``titles`` are still
        tagged (with ``data-fasta`` only).

    Returns
    -------
    str
        The SVG with ``data-tip``/``aria-label``/``data-fasta`` attributes added.
    """
    fasta_keys = fasta_keys or {}
    for gid in dict.fromkeys((*titles, *fasta_keys)):
        opening = f'<g id="{prefix}{gid}">'
        attrs = ''
        if gid in titles:
            esc = escape(titles[gid], quote=True)
            attrs += f' data-tip="{esc}" aria-label="{esc}"'
        if gid in fasta_keys:
            attrs += f' data-fasta="{escape(fasta_keys[gid], quote=True)}"'
        replacement = f'<g id="{prefix}{gid}"{attrs}>'
        svg = svg.replace(opening, replacement, 1)
    return svg


def _fasta_record(name, seq, width=60):
    """
    Format a name and sequence as a wrapped FASTA record string.

    Parameters
    ----------
    name : str
        The record identifier (the header after ``>``).
    seq : str
        The sequence; wrapped to ``width`` characters per line.
    width : int, optional
        Line-wrap width (default: 60).

    Returns
    -------
    str
        A FASTA record ending in a newline.
    """
    lines = [seq[i : i + width] for i in range(0, len(seq), width)] or ['']
    body = '\n'.join(lines)
    return f'>{name}\n{body}\n'


def _data_uri(text):
    """
    Encode text as a base64 ``data:`` URI for a self-contained download link.

    Parameters
    ----------
    text : str
        The payload (e.g. a FASTA document).

    Returns
    -------
    str
        A ``data:text/plain;charset=utf-8;base64,...`` URI.
    """
    b64 = base64.b64encode(text.encode('utf-8')).decode('ascii')
    return f'data:text/plain;charset=utf-8;base64,{b64}'


def _reference_phrase(label):
    """
    Describe the mutation-spectrum reference for the report prose.

    Parameters
    ----------
    label : str or None
        The reference sequence's id when a specific alignment row was chosen, or
        ``None`` to use the default deRIP-corrected consensus.

    Returns
    -------
    str
        An HTML phrase naming the reference (the label is code-formatted).
    """
    if label is None:
        return 'the reconstructed deRIP&rsquo;d ancestor'
    return f'reference sequence <code>{escape(str(label))}</code>'


def _zoom_control():
    """
    Build the (class-based) zoom control for a panel header.

    Returns
    -------
    str
        A ``<span class="zoom">`` with ``−`` / ``+`` buttons and a ``%`` label.
        Class-based (not id-based) so every panel can carry its own copy in its
        sticky header while the JS keeps them all in sync.
    """
    return (
        '<span class="zoom" title="Zoom the alignment and strand-bias figures">'
        '<button type="button" class="zoom-out">&minus;</button>'
        '<span class="zlabel">100%</span>'
        '<button type="button" class="zoom-in">+</button></span>'
    )


def _alignment_row_legend():
    """
    Build the HTML colour key for the alignment-row figure.

    Returns
    -------
    str
        A ``<div class="legend">`` naming the base colours and the triangle
        marker colours (product / substrate / non-RIP).
    """
    bases = ''.join(
        f'<span class="lg"><i style="background:{BASE_COLORS[b]}"></i>{b}</span>'
        for b in ('A', 'C', 'G', 'T')
    )
    markers = (
        f'<span class="lg"><b style="color:{PRODUCT_COLOR}">&#9660;</b> product</span>'
        f'<span class="lg"><b style="color:{SUBSTRATE_COLOR}">&#9660;</b> substrate</span>'
        f'<span class="lg"><b style="color:{NONRIP_COLOR}">&#9660;</b> '
        'non-RIP</span>'
    )
    return (
        '<div class="legend">'
        '<span class="lg-title">Bases</span>'
        + bases
        + '<span class="lg-title">Markers</span>'
        + markers
        + '</div>'
    )


def _stats_sections_html(row):
    """
    Render one sequence's statistics as grouped, transposed cards.

    Rather than a single wide row, the statistics are split into related
    sections (RIP events, strand bias, CRI, composition), each a small
    stat/value table with a short description of what the numbers mean.

    Parameters
    ----------
    row : pandas.Series
        One row of :meth:`derip2.derip.DeRIP.summarize_stats` (i.e.
        ``df.iloc[row_index]``).

    Returns
    -------
    str
        A ``<div class="stat-grid">`` of cards.
    """
    cards = []
    for title, description, fields in _STAT_SECTIONS:
        rows_html = []
        for column, label in fields:
            if column == 'RIP_total':
                # Derived: forward + reverse RIP events.
                total = int(row['RIP_fwd']) + int(row['RIP_rev'])
                text, css = str(total), ''
            else:
                text, css = _format_cell(column, row[column])
            # Flag a positive-RIP CRI (> 1) in green, as the RIP-signal threshold.
            if column == 'CRI' and float(row['CRI']) > 1:
                css = 'pos'
            # Bold a significant strand-asymmetry p-value.
            if column == 'pvalue' and float(row['pvalue']) < 0.05:
                css = (css + ' sig').strip()
            cls = f' {css}' if css else ''
            rows_html.append(
                f'<tr><th scope="row">{escape(label)}</th>'
                f'<td class="value{cls}">{text}</td></tr>'
            )
        cards.append(
            f'<div class="stat-card"><h4>{escape(title)}</h4>'
            f'<p class="desc">{escape(description)}</p>'
            f'<table><tbody>{"".join(rows_html)}</tbody></table></div>'
        )
    return '<div class="stat-grid">' + ''.join(cards) + '</div>'


# Extra CSS layered on top of the shared report style: the panel show/hide
# mechanism, the navigation bar, the horizontally-scrolling wide figures, and
# the transposed statistics grid. Kept theme-agnostic (it inherits the
# light/dark variables from ``_STYLE``).
_PSR_STYLE = """
.seq-panel[hidden] { display: none; }
.seq-nav {
  position: sticky; top: 0; z-index: 10; display: flex; align-items: center;
  gap: .75rem; padding: .6rem .9rem; margin: 0 0 1.4rem;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
}
.seq-nav button {
  font: inherit; font-size: 13px; padding: .3rem .8rem; cursor: pointer;
  background: var(--page); color: var(--ink); border: 1px solid var(--rule);
  border-radius: 6px;
}
.seq-nav button:hover { border-color: var(--muted); }
.seq-nav .indicator { font-variant-numeric: tabular-nums; color: var(--ink-2); }
.seq-nav .hint { color: var(--muted); font-size: 12px; margin-left: auto; }
/* Simultaneous zoom control for the column-aligned figures, now living in the
   sticky sequence header, right-aligned. */
.zoom { display: inline-flex; align-items: center; gap: .3rem; margin-left: auto; }
.zoom button {
  font: inherit; font-size: 13px; cursor: pointer; width: 1.9rem;
  text-align: center; padding: .25rem 0; font-weight: 600;
  background: var(--page); color: var(--ink); border: 1px solid var(--rule);
  border-radius: 6px;
}
.zoom button:hover { border-color: var(--muted); }
.zoom .zlabel {
  min-width: 3.2rem; text-align: center; font-variant-numeric: tabular-nums;
  color: var(--ink-2); font-size: 12px;
}
/* Keep the sequence header (number, name, length) + zoom pinned below the nav
   bar as the reader scrolls down a long panel. */
.seq-panel h2 {
  position: sticky; top: 2.9rem; z-index: 9; margin: 0 0 .6rem;
  padding: .5rem 0; background: var(--page); border-bottom: 1px solid var(--rule);
  display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
}
.seq-panel h2 .seqid { color: var(--ink-2); font-weight: 400; }
.seq-panel h2 .seqlen { color: var(--muted); font-weight: 400; font-size: 1rem; }
.seq-panel h3 {
  font-size: 1rem; font-weight: 600; margin: 1.6rem 0 .2rem;
  padding-top: .8rem; border-top: 1px solid var(--rule);
}
.desc { color: var(--ink-2); margin: .1rem 0 .8rem; max-width: 74ch; font-size: 14px; }
.note { color: var(--muted); font-size: 13px; }
/* Compact flank-context comparison table (5 rows): size columns to their
   content and never wrap a cell, so the long comparison names stay on one line. */
.flank-compare { width: auto; table-layout: auto; margin: .4rem 0; }
.flank-compare th, .flank-compare td { white-space: nowrap; }
.flank-compare td:not(:first-child), .flank-compare th:not(:first-child) {
  text-align: right; font-variant-numeric: tabular-nums;
}

/* Colour key for the alignment-row figure. */
.legend {
  display: flex; flex-wrap: wrap; align-items: center; gap: .35rem .9rem;
  margin: 0 0 .6rem; font-size: 12.5px; color: var(--ink-2);
}
.legend .lg-title { color: var(--muted); font-weight: 600; }
.legend .lg { display: inline-flex; align-items: center; gap: .3rem; }
.legend .lg i {
  display: inline-block; width: .8rem; height: .8rem; border-radius: 2px;
}
.legend .lg b { font-size: 1rem; line-height: 1; }

/* Significant strand-asymmetry p-value (< 0.05): coloured green, not bold. */
.stat-card td.value.sig { color: #007a3d; }

/* Wide figures (alignment row, strand bias) keep their intrinsic width and
   scroll horizontally rather than being squashed to page width, so bars stay
   readable on long alignments. */
.col-scroll {
  overflow-x: auto; background: #fcfcfb; border-radius: 6px; padding: .3rem;
}
.col-scroll svg { display: block; max-width: none; height: auto; }

/* The overview page's full alignment figure (inline SVG, vector chrome + one
   embedded raster grid) scrolls in BOTH axes and zooms with the shared control;
   capped height so a tall alignment does not run off the page before you scroll
   it. */
.aln-scroll {
  overflow: auto; max-height: 80vh; background: #fcfcfb; border-radius: 6px;
  padding: .3rem;
}
.aln-scroll svg { display: block; max-width: none; height: auto; }

/* The completion and GC bars are fixed-width; centre them and cap to the page
   so they stay aligned across sequences. */
.figure-fixed { overflow-x: auto; background: #fcfcfb; border-radius: 6px; padding: .5rem; }
.figure-fixed svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }

/* The spectra are wide (96 trinucleotide ticks). They keep their intrinsic
   width and scroll horizontally on their own, so the ticks never overlap; a
   fixed figure geometry keeps the plot body aligned across sequences. This
   scroll is independent of the alignment column figures. */
.spectrum-scroll { overflow-x: auto; background: #fcfcfb; border-radius: 6px; padding: .5rem; }
.spectrum-scroll svg { display: block; margin: 0 auto; max-width: none; height: auto; }

/* Transposed, grouped statistics: a responsive grid of small sections, each a
   two-column stat/value table with a short description. */
.stat-grid {
  display: grid; gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}
.stat-card {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 8px; padding: .8rem 1rem;
}
.stat-card h4 { margin: 0 0 .2rem; font-size: .95rem; font-weight: 600; }
.stat-card .desc { font-size: 12.5px; margin: 0 0 .6rem; }
.stat-card table { width: 100%; font-size: 13px; }
.stat-card th, .stat-card td {
  text-align: left; padding: .3rem .2rem; border-bottom: 1px solid var(--rule);
}
.stat-card td.value {
  text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;
}
.stat-card tr:last-child th, .stat-card tr:last-child td { border-bottom: none; }

/* Custom annotation tooltip: a single floating element positioned near the
   cursor by JS. Replaces native SVG <title> (which has a ~1 s delay and no
   click behaviour). Annotation groups carry a data-tip attribute. */
[data-tip] { cursor: pointer; }
.psr-tip {
  position: fixed; z-index: 50; pointer-events: none;
  max-width: 320px; padding: .3rem .5rem; border-radius: 6px;
  background: var(--ink); color: var(--surface);
  border: 1px solid var(--rule);
  font-size: 12.5px; line-height: 1.3; white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0, 0, 0, .25);
}
.psr-tip[hidden] { display: none; }

/* Overview download / view-FASTA toolbar. */
.psr-toolbar {
  display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 .8rem;
}
.psr-btn {
  font: inherit; font-size: 13px; padding: .35rem .8rem; cursor: pointer;
  background: var(--page); color: var(--ink); border: 1px solid var(--rule);
  border-radius: 6px; text-decoration: none; display: inline-block;
}
.psr-btn:hover { border-color: var(--muted); }

/* The overview annotation groups and the deRIP consensus row become clickable:
   pointer-events:all lets even the invisible consensus overlay catch a click. */
.aln-scroll [data-fasta], .aln-scroll [data-fasta] * {
  cursor: pointer; pointer-events: all;
}

/* Click-to-view FASTA modal: a centred dialog over a dimming backdrop. */
.psr-modal { position: fixed; inset: 0; z-index: 60; }
.psr-modal[hidden] { display: none; }
.psr-modal-backdrop {
  position: absolute; inset: 0; background: rgba(0, 0, 0, .45);
}
.psr-modal-box {
  position: relative; margin: 6vh auto 0; max-width: 680px; width: calc(100% - 2rem);
  max-height: 84vh; display: flex; flex-direction: column;
  background: var(--page); color: var(--ink);
  border: 1px solid var(--rule); border-radius: 10px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, .35);
}
.psr-modal-head {
  display: flex; align-items: center; gap: .5rem;
  padding: .7rem 1rem; border-bottom: 1px solid var(--rule);
}
.psr-modal-title { font-weight: 600; word-break: break-all; }
.psr-modal-x {
  margin-left: auto; font: inherit; font-size: 1.3rem; line-height: 1;
  background: none; border: none; color: var(--muted); cursor: pointer;
  padding: 0 .2rem;
}
.psr-modal-x:hover { color: var(--ink); }
.psr-tabs { display: flex; gap: .3rem; padding: .6rem 1rem 0; }
.psr-tab {
  font: inherit; font-size: 13px; padding: .3rem .8rem; cursor: pointer;
  background: var(--surface); color: var(--ink-2);
  border: 1px solid var(--rule); border-bottom: none;
  border-radius: 6px 6px 0 0;
}
.psr-tab.is-active { color: var(--ink); font-weight: 600; background: var(--page); }
.psr-tab[hidden] { display: none; }
.psr-fasta {
  margin: 0; padding: .8rem 1rem; overflow: auto; flex: 1 1 auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12.5px; line-height: 1.45; white-space: pre; word-break: normal;
  background: #fcfcfb; color: #111;
}
.psr-modal-foot {
  display: flex; align-items: center; gap: .8rem;
  padding: .6rem 1rem; border-top: 1px solid var(--rule);
}
.psr-transl-note { color: var(--muted); font-size: 12.5px; margin: 0; }
.psr-transl-note[hidden] { display: none; }
.psr-copy {
  font: inherit; font-size: 13px; padding: .3rem .9rem; cursor: pointer;
  background: var(--page); color: var(--ink); border: 1px solid var(--rule);
  border-radius: 6px;
}
.psr-copy:hover { border-color: var(--muted); }

/* Overview all-sequence summary statistics table (sortable). The box scrolls in
   both axes with a capped height; the two header rows stay pinned to the top and
   the sequence-name column stays pinned to the left (via position: sticky). The
   --h1 offset is the height of the first (group) header row, so the second header
   row sits directly below it. */
.stats-scroll {
  --h1: 1.9rem;
  overflow: auto; max-height: 70vh; margin: .2rem 0 1rem;
}
.psr-stats {
  border-collapse: separate; border-spacing: 0; font-size: 13px;
  white-space: nowrap; font-variant-numeric: tabular-nums;
}
.psr-stats th, .psr-stats td {
  padding: .35rem .6rem; border-bottom: 1px solid var(--rule); text-align: right;
}
.psr-stats thead th {
  background: var(--surface); position: sticky; z-index: 3;
  border-bottom: 1px solid var(--rule);
}
.psr-stats thead tr:first-child th { top: 0; height: var(--h1); }
.psr-stats thead tr:nth-child(2) th { top: var(--h1); }
.psr-stats thead th.grp {
  text-align: center; font-weight: 600;
  border-left: 1px solid var(--rule); border-right: 1px solid var(--rule);
}
/* Sticky first column (sequence names) — header corner sits above everything. */
.psr-stats th[scope="row"] {
  text-align: left; font-weight: 600; position: sticky; left: 0; z-index: 2;
  background: var(--page); border-right: 1px solid var(--rule);
}
.psr-stats thead th.corner { left: 0; z-index: 4; }
.psr-stats tbody tr:hover th[scope="row"] { background: var(--surface); }
.psr-stats th.sortable { cursor: pointer; user-select: none; }
.psr-stats th.sortable:hover { color: var(--ink); }
.psr-stats th.sortable::after { content: ''; margin-left: .3rem; color: var(--muted); }
.psr-stats th.sortable[data-dir="asc"]::after { content: '▲'; }
.psr-stats th.sortable[data-dir="desc"]::after { content: '▼'; }
.psr-stats .seq-link {
  color: inherit; text-decoration: underline; text-decoration-style: dotted;
  text-underline-offset: 2px; cursor: pointer;
}
.psr-stats .seq-link:hover { text-decoration-style: solid; }
.psr-stats td.value.muted { color: var(--muted); }
.psr-stats td.value.pos { color: #007a3d; }
.psr-stats td.value.neg { color: #b4292a; }
.psr-stats tbody tr:hover td { background: var(--surface); }
.psr-stats tr.consensus-row th[scope="row"],
.psr-stats tr.consensus-row td { font-weight: 600; }
.psr-stats tr.consensus-row td, .psr-stats tr.consensus-row th[scope="row"] {
  border-top: 2px solid var(--muted);
}
"""

# The click-to-view FASTA modal, injected once per report. Populated and shown by
# the popup handler in ``_PSR_SCRIPT``; ``data-close`` marks the backdrop and the
# × button so a single handler can dismiss it.
_MODAL_HTML = (
    '<div class="psr-modal" id="psr-modal" hidden>'
    '<div class="psr-modal-backdrop" data-close></div>'
    '<div class="psr-modal-box" role="dialog" aria-modal="true" '
    'aria-labelledby="psr-modal-title">'
    '<div class="psr-modal-head">'
    '<span class="psr-modal-title" id="psr-modal-title"></span>'
    '<button class="psr-modal-x" type="button" data-close aria-label="Close">'
    '&times;</button>'
    '</div>'
    '<div class="psr-tabs" id="psr-tabs">'
    '<button class="psr-tab is-active" type="button" data-tab="nt">Nucleotide</button>'
    '<button class="psr-tab" type="button" data-tab="aa">Translation</button>'
    '</div>'
    '<pre class="psr-fasta" id="psr-fasta"></pre>'
    '<div class="psr-modal-foot">'
    '<button class="psr-copy" id="psr-copy" type="button">Copy</button>'
    '<p class="psr-transl-note" id="psr-transl-note" hidden></p>'
    '</div>'
    '</div></div>'
)

# Dependency-free navigation. Beyond stepping between panels, it preserves both
# scroll axes so content stays aligned when flipping pages: the window's vertical
# scroll is never reset, and the horizontal scroll of the column-aligned figures
# (alignment row + strand bias, which share the same column axis) is remembered
# and re-applied to whichever panel is shown. Because every sequence has the same
# number of columns and the figures use a fixed geometry, a given scroll offset
# lands on the same column on every page.
_PSR_SCRIPT = """
(function () {
  var panels = Array.prototype.slice.call(
    document.querySelectorAll('.seq-panel'));
  if (!panels.length) return;
  var indicator = document.getElementById('seq-indicator');
  var current = 0;
  var savedLeft = 0;      // shared horizontal offset for the column figures
  var syncing = false;    // guard against scroll-event feedback while syncing

  function colScrollers(panel) {
    return Array.prototype.slice.call(panel.querySelectorAll('.col-scroll'));
  }

  // Page 0 is the alignment overview; the rest are sequences 1..N.
  var nSeqs = panels.length - 1;
  function show(k) {
    current = (k + panels.length) % panels.length;
    panels.forEach(function (p, i) {
      if (i === current) { p.removeAttribute('hidden'); }
      else { p.setAttribute('hidden', ''); }
    });
    if (indicator) {
      indicator.textContent = current === 0
        ? 'Overview'
        : 'Sequence ' + current + ' / ' + nSeqs;
    }
    // Re-apply the remembered horizontal offset; leave the vertical scroll be.
    syncing = true;
    colScrollers(panels[current]).forEach(function (el) { el.scrollLeft = savedLeft; });
    syncing = false;
  }

  // Remember the horizontal offset whenever the user scrolls a column figure,
  // and mirror it to the other column figures in the same panel.
  panels.forEach(function (panel) {
    colScrollers(panel).forEach(function (el) {
      el.addEventListener('scroll', function () {
        if (syncing) return;
        savedLeft = el.scrollLeft;
        syncing = true;
        colScrollers(panel).forEach(function (other) {
          if (other !== el) { other.scrollLeft = savedLeft; }
        });
        syncing = false;
      });
    });
  });

  document.getElementById('seq-prev').addEventListener('click', function () {
    show(current - 1);
  });
  document.getElementById('seq-next').addEventListener('click', function () {
    show(current + 1);
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft') { show(current - 1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { show(current + 1); e.preventDefault(); }
  });

  // Simultaneous zoom for every column-aligned figure (alignment row + strand
  // bias) and the overview alignment image, across all panels, so they scale
  // together. Each panel carries its own zoom control (in its sticky header);
  // the controls are class-based and kept in sync. Base pixel width comes from
  // an SVG's point size (1pt = 4/3 px) or an image's natural width.
  var zoom = 1;
  // Both the column strips and the overview are inline SVG; scale them all by
  // their intrinsic point width (1pt = 4/3 px).
  var zoomSvgs = Array.prototype.slice.call(
    document.querySelectorAll('.col-scroll svg, .aln-scroll svg'));
  function svgBasePx(svg) {
    var w = parseFloat(svg.getAttribute('width') || '0');
    return w * 4 / 3;  // pt -> css px
  }
  function applyZoom() {
    zoomSvgs.forEach(function (svg) {
      svg.style.width = (svgBasePx(svg) * zoom) + 'px';
    });
    document.querySelectorAll('.zlabel').forEach(function (l) {
      l.textContent = Math.round(zoom * 100) + '%';
    });
  }
  document.querySelectorAll('.zoom-in').forEach(function (b) {
    b.addEventListener('click', function () {
      zoom = Math.min(zoom * 1.25, 8); applyZoom();
    });
  });
  document.querySelectorAll('.zoom-out').forEach(function (b) {
    b.addEventListener('click', function () {
      zoom = Math.max(zoom / 1.25, 0.25); applyZoom();
    });
  });
  applyZoom();

  // Custom annotation tooltip: show the hovered group's data-tip with no delay,
  // follow the cursor, and pin it on click (the only path on touch devices).
  var tip = document.getElementById('psr-tip');
  var pinned = false;
  function placeTip(e) {
    var pad = 12;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > window.innerWidth) { x = e.clientX - w - pad; }
    if (y + h > window.innerHeight) { y = e.clientY - h - pad; }
    tip.style.left = Math.max(0, x) + 'px';
    tip.style.top = Math.max(0, y) + 'px';
  }
  function showTip(text, e) {
    tip.textContent = text;
    tip.removeAttribute('hidden');
    placeTip(e);
  }
  function hideTip() {
    if (pinned) return;
    tip.setAttribute('hidden', '');
  }
  if (tip) {
    document.addEventListener('mouseover', function (e) {
      if (pinned) return;
      var g = e.target.closest && e.target.closest('[data-tip]');
      if (g) { showTip(g.getAttribute('data-tip'), e); }
    });
    document.addEventListener('mousemove', function (e) {
      if (pinned || tip.hasAttribute('hidden')) return;
      var g = e.target.closest && e.target.closest('[data-tip]');
      if (g) { placeTip(e); } else { tip.setAttribute('hidden', ''); }
    });
    document.addEventListener('mouseout', function (e) {
      if (pinned) return;
      var g = e.target.closest && e.target.closest('[data-tip]');
      if (g) { hideTip(); }
    });
    document.addEventListener('click', function (e) {
      // A group with a FASTA payload opens the popup instead of pinning a tip.
      var fa = e.target.closest && e.target.closest('[data-fasta]');
      if (fa) {
        openFastaModal(fa.getAttribute('data-fasta'));
        pinned = false; tip.setAttribute('hidden', '');
        return;
      }
      var g = e.target.closest && e.target.closest('[data-tip]');
      if (g) {
        pinned = true; showTip(g.getAttribute('data-tip'), e);
      } else {
        pinned = false; tip.setAttribute('hidden', '');
      }
    });
  }

  // Click-to-view FASTA popup. The payloads are embedded as JSON; each clickable
  // group (a CDS annotation, or the deRIP consensus row) carries a data-fasta key.
  var fastaData = {};
  var dataEl = document.getElementById('psr-fasta-data');
  if (dataEl) { try { fastaData = JSON.parse(dataEl.textContent); } catch (err) {} }
  var modal = document.getElementById('psr-modal');
  var modalTitle = document.getElementById('psr-modal-title');
  var fastaPre = document.getElementById('psr-fasta');
  var translNote = document.getElementById('psr-transl-note');
  var copyBtn = document.getElementById('psr-copy');
  var modalTabs = modal
    ? Array.prototype.slice.call(modal.querySelectorAll('.psr-tab')) : [];
  var fastaCurrent = null;   // the active payload
  var activeTab = 'nt';

  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (err) {}
    document.body.removeChild(ta);
  }

  function renderFastaTab() {
    if (!fastaCurrent) return;
    var aaTab = modalTabs.filter(function (t) {
      return t.getAttribute('data-tab') === 'aa';
    })[0];
    var hasAa = !!fastaCurrent.aa;
    if (aaTab) {
      if (hasAa) { aaTab.removeAttribute('hidden'); }
      else { aaTab.setAttribute('hidden', ''); }
    }
    if (activeTab === 'aa' && !hasAa) { activeTab = 'nt'; }
    modalTabs.forEach(function (t) {
      t.classList.toggle('is-active', t.getAttribute('data-tab') === activeTab);
    });
    fastaPre.textContent = activeTab === 'aa' ? fastaCurrent.aa : fastaCurrent.nt;
    if (activeTab === 'aa' && fastaCurrent.table != null) {
      translNote.textContent =
        'Translation — NCBI genetic code table ' + fastaCurrent.table + '.';
      translNote.removeAttribute('hidden');
    } else {
      translNote.setAttribute('hidden', '');
    }
  }

  function openFastaModal(key) {
    if (!modal || !fastaData[key]) return;
    fastaCurrent = fastaData[key];
    activeTab = 'nt';
    modalTitle.textContent = fastaCurrent.name;
    renderFastaTab();
    modal.removeAttribute('hidden');
  }
  function closeFastaModal() {
    if (modal) { modal.setAttribute('hidden', ''); }
    fastaCurrent = null;
  }

  if (modal) {
    modalTabs.forEach(function (t) {
      t.addEventListener('click', function () {
        activeTab = t.getAttribute('data-tab'); renderFastaTab();
      });
    });
    modal.addEventListener('click', function (e) {
      if (e.target.closest('[data-close]')) { closeFastaModal(); }
    });
    copyBtn.addEventListener('click', function () {
      var text = fastaPre.textContent;
      function done() {
        var old = copyBtn.textContent; copyBtn.textContent = 'Copied!';
        setTimeout(function () { copyBtn.textContent = old; }, 1200);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {
          fallbackCopy(text); done();
        });
      } else { fallbackCopy(text); done(); }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && !modal.hasAttribute('hidden')) { closeFastaModal(); }
    });
  }

  // Sortable overview stats table. Each sortable header carries its column index
  // (data-ci); a click sorts the tbody rows by that column. Cells are compared
  // numerically when both parse as numbers (a leading '+' is stripped), otherwise
  // as text; en-dash / empty cells (not-applicable stats) always sort last.
  function cellVal(td) {
    var t = (td.textContent || '').trim();
    if (t === '' || t === '\\u2013' || t === '\\u2014') { return { n: null, s: '' }; }
    var n = parseFloat(t.replace('+', ''));
    return isNaN(n) ? { n: null, s: t } : { n: n, s: t };
  }
  var statsTable = document.querySelector('table.psr-stats');
  if (statsTable && statsTable.tBodies.length) {
    var statsBody = statsTable.tBodies[0];
    var sortDir = null, sortCi = null;
    Array.prototype.slice.call(
      statsTable.querySelectorAll('th.sortable')).forEach(function (th) {
      th.addEventListener('click', function () {
        var ci = parseInt(th.getAttribute('data-ci'), 10);
        var asc = !(sortCi === ci && sortDir === 'asc');
        sortCi = ci; sortDir = asc ? 'asc' : 'desc';
        var rows = Array.prototype.slice.call(statsBody.rows);
        rows.sort(function (a, b) {
          var x = cellVal(a.cells[ci]), y = cellVal(b.cells[ci]);
          if (x.n === null && y.n === null) {
            return asc ? x.s.localeCompare(y.s) : y.s.localeCompare(x.s);
          }
          if (x.n === null) { return 1; }   // not-applicable sorts last
          if (y.n === null) { return -1; }
          return asc ? x.n - y.n : y.n - x.n;
        });
        rows.forEach(function (r) { statsBody.appendChild(r); });
        Array.prototype.slice.call(
          statsTable.querySelectorAll('th.sortable')).forEach(function (h) {
          h.removeAttribute('data-dir');
        });
        th.setAttribute('data-dir', asc ? 'asc' : 'desc');
      });
    });
  }

  // Sequence names in the overview stats table jump to that sequence's page.
  // Scroll position is preserved (as with arrow-key navigation); the sticky nav
  // and panel header keep the reader oriented.
  document.addEventListener('click', function (e) {
    var link = e.target.closest && e.target.closest('[data-goto]');
    if (!link) { return; }
    e.preventDefault();
    show(parseInt(link.getAttribute('data-goto'), 10));
  });

  // Land on the overview with the whole MSA figure visible: shrink the shared
  // zoom just enough to fit the alignment figure in its scroll box (never zoom in
  // past 100%). Widths are then applied by applyZoom below.
  function fitOverview() {
    var box = document.querySelector(
      '.seq-panel[data-index="overview"] .aln-scroll');
    if (!box) { return; }
    var svg = box.querySelector('svg');
    if (!svg) { return; }
    var base = svgBasePx(svg);
    var avail = box.clientWidth - 12;   // minus the box padding
    if (base > 0 && avail > 0 && avail < base) {
      zoom = Math.max(0.25, avail / base);
    }
  }

  show(0);
  fitOverview();
  applyZoom();
})();
"""


def _select_rows(df, max_seqs):
    """
    Choose which sequence rows to render, and whether the report is truncated.

    When ``max_seqs`` caps the report below the alignment size, the sequences
    with the strongest strand bias (largest ``|RSI|``) are kept, so the most
    informative panels survive the cap.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :meth:`derip2.derip.DeRIP.summarize_stats`, one row per
        sequence in alignment order.
    max_seqs : int or None
        Maximum number of sequences to render. ``None`` renders all.

    Returns
    -------
    tuple
        ``(indices, truncated)``: a list of alignment row indices to render (in
        alignment order) and a bool indicating whether any sequences were
        dropped.
    """
    n = len(df)
    if max_seqs is None or n <= max_seqs:
        return list(range(n)), False

    # Rank by |RSI|; NaN RSI (undefined strand) sorts last. Keep the top
    # ``max_seqs`` but present them back in alignment order.
    ranked = df['RSI'].abs().fillna(-1.0).sort_values(ascending=False)
    kept = sorted(ranked.index[:max_seqs].tolist())
    return kept, True


_EFFECT_COLUMNS = (
    ('kind', 'Effect'),
    ('aa_pos', 'AA position'),
    ('ref_aa', 'Ancestral'),
    ('alt_aa', 'Observed'),
    ('gapped_col', 'Column'),
    ('nt', 'Nucleotide'),
)


def _effects_table_html(effects, deripd_aa):
    """
    Render a sequence's gene-effect records and restored translations as HTML.

    Parameters
    ----------
    effects : list of derip2.annotation.EffectRecord
        The effects for one sequence.
    deripd_aa : dict of str to str
        Per-gene deRIP'd (restored) protein strings for genes on this sequence.

    Returns
    -------
    str
        A ``<table>`` of effects followed by the restored translations, or a
        note when the sequence has a gene but no RIP-induced effect.
    """
    head = ''.join(
        f'<th scope="col">{escape(label)}</th>' for _key, label in _EFFECT_COLUMNS
    )

    rows = []
    for effect in effects:
        nt = (
            f'{effect.nt_ref or ""}&rarr;{effect.nt_alt or ""}'
            if (effect.nt_ref or effect.nt_alt)
            else '&ndash;'
        )
        values = {
            'kind': escape(effect.kind),
            'aa_pos': '&ndash;' if effect.aa_pos is None else str(effect.aa_pos),
            'ref_aa': escape(effect.ref_aa or '&ndash;'),
            'alt_aa': escape(effect.alt_aa or '&ndash;'),
            'gapped_col': '&ndash;'
            if effect.gapped_col is None
            else str(effect.gapped_col),
            'nt': nt,
        }
        cells = ''.join(f'<td>{values[key]}</td>' for key, _label in _EFFECT_COLUMNS)
        rows.append(f'<tr>{cells}</tr>')

    table = ''
    if rows:
        table = (
            '<div class="table-wrap"><table><thead><tr>'
            + head
            + '</tr></thead><tbody>'
            + ''.join(rows)
            + '</tbody></table></div>'
        )
    else:
        table = '<p class="note">No RIP-induced coding change in this sequence.</p>'

    aa_blocks = ''.join(
        f'<p class="note"><code>{escape(gene_id)}</code> deRIP-restored protein: '
        f'<code>{escape(aa)}</code></p>'
        for gene_id, aa in sorted(deripd_aa.items())
    )
    return table + aa_blocks


def _panel_html(
    derip,
    df,
    spectra,
    downstream,
    flank,
    flank_comparisons,
    row_index,
    panel_number,
    effects_by_seq,
    genes_by_seqid,
    deripd_aa,
    cds_gene_cols=(),
    genetic_code=1,
    spectra_ref_index=None,
    spectra_ref_label=None,
):
    """
    Build the HTML for one sequence's panel.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        The analysed DeRIP object.
    df : pandas.DataFrame
        The per-sequence statistics table.
    spectra : derip2.stats.mutation_spectra.SpectraResult
        Per-row trinucleotide SBS-96 spectra (one sample column per sequence).
    downstream : derip2.stats.mutation_spectra.SpectraResult
        Per-row downstream-triplet spectra (one sample column per sequence).
    flank : derip2.stats.flank_spectra.FlankSpectraResult
        Per-row flanking-context spectra of RIP-like sites.
    flank_comparisons : dict of str to dict
        The five flank-context comparisons for this row (see
        :func:`derip2.stats.flank_spectra.compare_flank_spectra`).
    row_index : int
        Alignment row index of the sequence.
    panel_number : int
        1-based position of this panel among those rendered (for the heading).
    effects_by_seq : dict of str to list of derip2.annotation.EffectRecord
        Per-sequence gene effects; empty when no GFF was supplied.
    genes_by_seqid : dict of str to list of derip2.annotation.Gene
        Genes keyed by sequence identifier; empty when no GFF was supplied.
    deripd_aa : dict of str to str
        Per-gene deRIP'd translations, keyed by gene identifier.
    cds_gene_cols : sequence of tuple, optional
        ``(gene, cds_columns, exon_spans, colour)`` with each gene's CDS
        projected onto the shared alignment columns; used to draw the
        per-subject CDS track. Empty when no GFF was supplied.
    genetic_code : int, optional
        NCBI translation table for the projected stop-codon calls (default: 1).
    spectra_ref_index : int, optional
        Alignment row index of the sequence used as the spectra reference; when
        it equals ``row_index`` this panel's spectrum is a self-comparison
        (empty by definition) and a note is shown. ``None`` = deRIP consensus.
    spectra_ref_label : str, optional
        The reference sequence's id, used in the spectrum prose. ``None`` = the
        default deRIP-corrected consensus.

    Returns
    -------
    str
        The ``<section>`` element for this sequence.
    """
    import matplotlib.pyplot as plt

    from derip2.plotting.persequence import (
        gc_content_bar,
        per_sequence_strand_bias,
        rip_completion_bar,
        sequence_row_strip,
    )
    from derip2.plotting.spectra import plot_downstream, plot_sbs96

    cls = derip.column_classes
    seq_id = derip.alignment[row_index].id
    consensus_seq = str(derip.gapped_consensus.seq)
    row_stats = df.iloc[row_index]
    n_cols = cls.arr.shape[1]
    # Ungapped length: the count of non-gap bases in this row.
    ungapped_len = int((cls.arr[row_index] != b'-').sum())

    # Wide figures share one width and identical fixed margins so the column axis
    # lands on the same pixels in both (aligning the alignment row with the bias
    # strip) and on the same pixels on every page (so a horizontal scroll offset
    # points at the same column across sequences). ~0.09 in/column keeps bars and
    # base cells legible; capped so the inline SVG stays a sane size on long
    # alignments. The zoom control scales both figures further, together.
    wide_w = max(6.0, min(360.0, n_cols * 0.09))

    # Project each gene's CDS onto this subject: same alignment columns for all
    # sequences, but the stop codons are this subject's own (RIP often adds
    # premature stops). One track band per gene, drawn below the deRIP row.
    cds_tracks = []
    if cds_gene_cols:
        from derip2.annotation import cds_display_id, cds_stop_columns

        for gene, cols, exon_spans, colour in cds_gene_cols:
            stops = cds_stop_columns(
                gene, cls.arr[row_index], cols, genetic_code=genetic_code
            )
            cds_tracks.append(
                (
                    exon_spans,
                    gene.strand,
                    stops,
                    gene.gene_id,
                    colour,
                    cds_display_id(gene),
                )
            )

    strip = sequence_row_strip(
        cls,
        row_index,
        seq_id=seq_id,
        consensus_seq=consensus_seq,
        cds_tracks=cds_tracks,
        width=wide_w,
        height=2.0 + 0.75 * len(cds_tracks),
    )
    _fix_wide_axes(strip, wide_w, n_cols, top_in=0.3, bottom_in=0.4)
    strip_svg = _figure_to_svg(strip, f's{row_index}row-', tight=False)
    strip_svg = _inject_svg_tooltips(
        strip_svg, f's{row_index}row-', getattr(strip, 'annotation_titles', {})
    )
    plt.close(strip)

    bias = per_sequence_strand_bias(cls, row_index, seq_id=seq_id, width=wide_w)
    _fix_wide_axes(bias, wide_w, n_cols, top_in=0.5, bottom_in=0.55)
    bias_svg = _figure_to_svg(bias, f's{row_index}bias-', tight=False)
    plt.close(bias)

    completion = rip_completion_bar(row_stats)
    completion_svg = _figure_to_svg(completion, f's{row_index}rip-', tight=False)
    plt.close(completion)

    gc = gc_content_bar(row_stats)
    gc_svg = _figure_to_svg(gc, f's{row_index}gc-', tight=False)
    plt.close(gc)

    # Wide, bare spectra (no redundant sample title / caption); fixed geometry so
    # the plot body aligns across pages, scrolled independently of the columns.
    # A touch narrower than the page so the scroll box can centre them.
    sbs = plot_sbs96(spectra, sample=row_index, width=11.0, bare=True)
    _fix_spectrum_axes(sbs)
    sbs_svg = _figure_to_svg(sbs, f's{row_index}sbs-', tight=False)
    plt.close(sbs)

    ds = plot_downstream(downstream, sample=row_index, width=11.0, bare=True)
    _fix_spectrum_axes(ds)
    ds_svg = _figure_to_svg(ds, f's{row_index}ds-', tight=False)
    plt.close(ds)

    # The three flank-context bihistograms (substrate left vs product right, one
    # per strand) as one figure, so a single unique id-prefix keeps the SVG glyph
    # ids collision-free.
    from derip2.plotting.flank_spectra import plot_flank_bihistograms

    flank_fig = plot_flank_bihistograms(flank, sample=row_index, bare=True)
    flank_svg = _figure_to_svg(flank_fig, f's{row_index}flank-', tight=True)
    plt.close(flank_fig)
    flank_table = _flank_comparison_table_html(flank_comparisons)

    # When this sequence is itself the chosen spectra reference, its spectrum is a
    # self-comparison and therefore empty by construction; flag that up front.
    ref_note = ''
    if spectra_ref_index is not None and row_index == spectra_ref_index:
        ref_note = (
            '<p class="note">This sequence is the chosen spectra reference, so its '
            'spectrum is empty (every base matches itself).</p>'
        )

    # Gene-effect panel: only shown when a GFF annotated this sequence.
    effect_html = ''
    if seq_id in genes_by_seqid:
        genes_here = {
            gene.gene_id: deripd_aa[gene.gene_id]
            for gene in genes_by_seqid[seq_id]
            if gene.gene_id in deripd_aa
        }
        effect_html = (
            '<h3>CDS SNP effects</h3>'
            '<p class="desc">Effect of this sequence’s RIP substitutions on the '
            'annotated coding sequence, predicted against the reconstructed '
            'ancestor, followed by the deRIP-restored protein.</p>'
        ) + _effects_table_html(effects_by_seq.get(seq_id, []), genes_here)

    return (
        f'<section class="seq-panel" data-index="{panel_number - 1}" hidden>'
        f'<h2>Sequence {panel_number}: <span class="seqid">{escape(seq_id)}</span> '
        f'<span class="seqlen">({ungapped_len} nt)</span>{_zoom_control()}</h2>'
        '<h3>Alignment row</h3>'
        '<p class="desc">The subject sequence (top) and the reconstructed deRIP’d '
        'reference (below), coloured by base identity; subject bases that match '
        'the reference are faded so the mismatches stand out. Triangle markers '
        'above the subject mark its role at each RIP-informative column, and '
        'RIP-like columns are shaded grey, as in the alignment-wide plot. When a '
        'gene model is supplied, a sub-plot below shows each CDS as rounded '
        'segments (yellow) joined across introns, with an arrowhead giving the '
        'strand, and a bold red <code>*</code> above the track at each stop codon '
        'in this sequence’s projected reading frame.</p>'
        f'{_alignment_row_legend()}'
        f'<div class="col-scroll">{strip_svg}</div>'
        '<h3>Per-sequence strand bias</h3>'
        '<p class="desc">One bar per RIP-like column this sequence takes part in: '
        'forward-strand events above the axis, reverse-strand events below. Bars '
        'are coloured by role — orange for the RIP product (TA), blue for the '
        'surviving substrate (CA on the forward strand, TG on the reverse). '
        'Scroll horizontally to follow long alignments.</p>'
        f'<div class="col-scroll">{bias_svg}</div>'
        '<h3>RIP completion</h3>'
        '<p class="desc">The fraction of this sequence’s available RIP-like sites '
        '(surviving substrate plus product) that have been converted to product, '
        'per strand and combined.</p>'
        f'<div class="figure-fixed">{completion_svg}</div>'
        '<h3>GC content</h3>'
        '<p class="desc">Base composition of this sequence. RIP lowers GC by '
        'converting C to T, so a low bar is consistent with heavy RIP.</p>'
        f'<div class="figure-fixed">{gc_svg}</div>'
        '<h3>Mutation spectrum (SBS-96)</h3>'
        f'{ref_note}'
        '<p class="desc">The single-base-substitution spectrum of this sequence '
        f'measured against {_reference_phrase(spectra_ref_label)}, in '
        'trinucleotide context (5′-N[R&gt;A]N-3′). RIP shows up as a C&gt;T peak '
        'in CpA context. Scroll horizontally if the 96 channels do not fit.</p>'
        f'<div class="spectrum-scroll">{sbs_svg}</div>'
        '<h3>Mutation spectrum (downstream context)</h3>'
        '<p class="desc">The same substitutions classified by the mutated base '
        'plus its two downstream bases (pyrimidine-folded), which resolves the '
        'CHG-methylation signal C&gt;T in CpNpG context.</p>'
        f'<div class="spectrum-scroll">{ds_svg}</div>'
        '<h3>Flanking-context spectra of RIP-like sites</h3>'
        '<p class="desc">For every RIP-like dinucleotide this sequence carries, the '
        'single base 1&nbsp;bp upstream and 1&nbsp;bp downstream is tallied as a '
        '4&nbsp;bp motif (the two centre bases fixed, the flanks varying &rarr; 16 '
        'channels). Each strand view is a <b>bihistogram</b>: surviving '
        '<b>substrate</b> counts (CpA forward / TpG reverse, counted anywhere) '
        'extend left and realised RIP <b>product</b> counts (TpA in RIP-informative '
        'columns) extend right, sharing a centre line. Reverse-strand motifs are '
        'reverse-complemented onto the CpA/TpA strand and every row is labelled on '
        'the left by its <b>CA-state</b> (substrate) motif and on the right by the '
        'equivalent <b>TA-state</b> (product) motif (e.g. <code>GCAG</code> '
        '&equiv; <code>GTAG</code>). A motif is marked '
        '<span style="color:#e34948">*</span> when its enrichment differs '
        'significantly between the two states: for each of the 16 flank contexts '
        'the substrate and product counts form one row of a 16&times;2 table, and '
        'that cell&rsquo;s <b>adjusted standardised (Haberman) residual</b> is '
        'tested against the standard normal &mdash; the motif is flagged when '
        '|z|&nbsp;&ge;&nbsp;1.96 (two-sided <i>p</i>&nbsp;&lt;&nbsp;0.05), provided '
        'both states have at least 20 sites, with no multiple-testing correction. '
        'The table below tests the same substrate-vs-product question overall '
        '(via the &chi;&sup2; homogeneity of the whole 16-channel spectra), and '
        'whether the two strands differ.</p>'
        f'<div class="spectrum-scroll">{flank_svg}</div>'
        f'{flank_table}'
        '<h3>Summary statistics</h3>'
        f'{_stats_sections_html(row_stats)}'
        f'{effect_html}'
        f'</section>'
    )


def _overview_svg(derip, cds_tracks, fasta_data=None):
    """
    Render the full ``--plot`` alignment figure and return it as inline SVG.

    Reuses :meth:`derip2.derip.DeRIP.plot_alignment` (the same figure the
    ``--plot`` flag writes, including the deRIP corrected consensus row). The
    figure is rendered to inline SVG so the axes, annotation track and consensus
    stay crisp vector (the coloured base grid remains a single embedded raster);
    the annotation groups carry ``data-tip`` attributes for the report tooltips
    and, where a FASTA payload exists, a ``data-fasta`` attribute so a click opens
    the sequence popup.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        The analysed DeRIP object.
    cds_tracks : list or None
        Rich per-gene CDS tracks ``(exon_spans, strand, stop_columns, label,
        colour, cds_id)`` to draw below the consensus (``--gff``), or None.
    fasta_data : dict, optional
        FASTA popup payloads keyed by CDS id (plus ``'__derip__'``). Used to add
        ``data-fasta`` attributes to the matching annotation groups and the
        consensus row.

    Returns
    -------
    str
        The inline ``<svg>`` fragment (id-prefixed, tooltip/fasta-tagged).
    """
    import matplotlib.pyplot as plt

    ali_height = len(derip.alignment)
    ali_length = derip.alignment.get_alignment_length()
    fig = derip.plot_alignment(
        return_figure=True,
        dpi=110,
        title=None,
        show_chars=(ali_height <= 25),
        draw_boxes=(ali_height <= 25),
        flag_corrected=(ali_length < 200),
        cds_tracks=cds_tracks,
    )
    try:
        titles = getattr(fig, 'annotation_titles', {})
        # Map each clickable group's gid to its FASTA-payload key: the consensus
        # row (gid 'deripseq') opens the deRIP sequence; each CDS exon group opens
        # its CDS (the tooltip text is 'cdsID — CDS exon x/y', so its id prefixes
        # the text). Only groups with a payload become clickable.
        valid = set(fasta_data or ())
        fasta_keys = {}
        for gid, text in titles.items():
            if gid == 'deripseq':
                if '__derip__' in valid:
                    fasta_keys[gid] = '__derip__'
            else:
                cid = text.split(' — ', 1)[0]
                if cid in valid:
                    fasta_keys[gid] = cid
        svg = _figure_to_svg(fig, 'ovw-', tight=True)
        svg = _inject_svg_tooltips(svg, 'ovw-', titles, fasta_keys)
    finally:
        plt.close(fig)
    return svg


def _overview_stats_table_html(df, derip, row_to_panel=None):
    """
    Build the sortable all-sequence statistics table for the overview page.

    One row per input sequence plus a final row for the deRIP-corrected
    consensus, with the columns grouped exactly as the per-sequence stat cards
    (:data:`_STAT_SECTIONS`). The consensus is the RIP-free reconstructed
    ancestor, so its RIP-event and strand-bias (RSI) columns are not applicable
    and render as an en-dash; only composition (GC) and the Composite RIP Index
    (CRI/PI/SI) are computed for it. A positive-RIP CRI (> 1) is coloured green
    and a significant strand-asymmetry p-value (< 0.05) is coloured green, as on
    the per-sequence cards. Every column is click-to-sort in the browser
    (numeric-aware; en-dash cells sort last).

    Parameters
    ----------
    df : pandas.DataFrame
        The per-sequence statistics (:meth:`derip2.derip.DeRIP.summarize_stats`).
    derip : derip2.derip.DeRIP
        The analysed DeRIP object (for the consensus row's GC and CRI).
    row_to_panel : dict of int to int, optional
        Maps an alignment row index to the 1-based panel position of that
        sequence's per-sequence page. Sequences present here get their name
        linked to their page; sequences dropped by ``--max-report-seqs`` (absent
        from the map) render as plain text.

    Returns
    -------
    str
        The ``<table class="psr-stats">`` markup (with its wrapping scroll div).
    """
    from Bio.SeqUtils import gc_fraction

    row_to_panel = row_to_panel or {}

    # Flatten the grouped layout into one ordered column list plus the group
    # spans that head it. 'RIP_total' is derived (fwd + rev), as on the cards.
    flat = []  # (column, label)
    group_spans = []  # (group title, colspan)
    for title, _desc, cols in _STAT_SECTIONS:
        group_spans.append((title, len(cols)))
        flat.extend(cols)

    # Two-row header: group titles spanning their columns, then the stat labels.
    grp_ths = ''.join(
        f'<th class="grp" colspan="{span}">{escape(title)}</th>'
        for title, span in group_spans
    )
    col_ths = ''.join(
        f'<th class="sortable" data-ci="{i + 1}">{escape(label)}</th>'
        for i, (_col, label) in enumerate(flat)
    )
    thead = (
        '<thead>'
        '<tr><th class="sortable corner" data-ci="0" rowspan="2">Sequence</th>'
        f'{grp_ths}</tr>'
        f'<tr>{col_ths}</tr>'
        '</thead>'
    )

    def _row_html(label, values, is_consensus=False, panel=None):
        """
        Build one table row: a leading label cell then a cell per flat column.

        Parameters
        ----------
        label : str
            The row header (a sequence id or the consensus id).
        values : dict
            Column-name to value; missing columns render as an en-dash.
        is_consensus : bool, optional
            Whether this is the deRIP-consensus row (adds a marker class).
        panel : int, optional
            1-based panel position of this sequence's per-sequence page; when
            given the name links to it. ``None`` leaves the name as plain text.

        Returns
        -------
        str
            The ``<tr>`` element for this row.
        """
        name = escape(str(label))
        if panel is not None:
            name = f'<a class="seq-link" href="#" data-goto="{panel}">{name}</a>'
        cells = [f'<th scope="row">{name}</th>']
        for col, _lab in flat:
            value = values.get(col)
            if col == 'RIP_total' and value is not None:
                text, css = str(int(value)), ''
            else:
                text, css = _format_cell(col, value)
            # Green flags, matching the per-sequence cards: a positive-RIP CRI
            # (> 1) and a significant strand-asymmetry p-value (< 0.05).
            if col == 'CRI' and isinstance(value, (int, float)) and value > 1:
                css = 'pos'
            elif col == 'pvalue' and isinstance(value, (int, float)) and value < 0.05:
                css = 'pos'
            cls = f' class="value {css}"'.rstrip() if css else ' class="value"'
            cells.append(f'<td{cls}>{text}</td>')
        tr_cls = ' class="consensus-row"' if is_consensus else ''
        return f'<tr{tr_cls}>{"".join(cells)}</tr>'

    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        values = {col: row[col] for col, _lab in flat if col in row.index}
        values['RIP_total'] = int(row['RIP_fwd']) + int(row['RIP_rev'])
        rows.append(_row_html(row['ID'], values, panel=row_to_panel.get(i)))

    # The deRIP consensus row: GC + CRI/PI/SI only; RIP/RSI columns stay None so
    # _format_cell renders them as an en-dash (not applicable to the ancestor).
    consensus_seq = derip.get_consensus_string()
    cri, pi, si = derip.calculate_cri(consensus_seq)
    consensus_values = {
        'GC': gc_fraction(consensus_seq) * 100,
        'CRI': cri,
        'PI': pi,
        'SI': si,
    }
    rows.append(_row_html(derip.consensus.id, consensus_values, is_consensus=True))

    return (
        '<div class="stats-scroll">'
        f'<table class="psr-stats">{thead}<tbody>{"".join(rows)}</tbody></table>'
        '</div>'
    )


def _overview_spectrum_svg(derip, ancestor=None):
    """
    Render the pooled SBS-96 spectrum (all sequences vs the spectra reference).

    Every alignment cell that differs from the reference is one substitution
    event; pooling all sequences into a single sample gives the alignment-wide
    mutation spectrum (dominated by the C→T / G→A RIP signature). Rendered like
    the per-sequence spectra (wide, bare, fixed geometry) so it scrolls on its own
    and reads consistently.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        The analysed DeRIP object.
    ancestor : str or None, optional
        Reference sequence (one base per alignment column) to compare every
        sequence against. ``None`` (default) uses the deRIP-corrected consensus.

    Returns
    -------
    str
        The inline ``<svg>`` fragment (id-prefixed), or ``''`` if no substitution
        events were observed.
    """
    import matplotlib.pyplot as plt

    from derip2.plotting.spectra import plot_sbs96

    spectra_all = derip.calculate_spectra(partition_by='none', ancestor=ancestor)
    fig = plot_sbs96(spectra_all, sample=0, width=11.0, bare=True)
    _fix_spectrum_axes(fig)
    svg = _figure_to_svg(fig, 'ovwsbs-', tight=False)
    plt.close(fig)
    return svg


# Human-readable names for the five flank-context comparisons, in display order.
_FLANK_COMPARISON_NAMES = {
    'sub_vs_prod_combined': 'Substrate vs product (combined)',
    'sub_vs_prod_fwd': 'Substrate vs product (forward)',
    'sub_vs_prod_rev': 'Substrate vs product (reverse)',
    'fwd_vs_rev_substrate': 'Forward vs reverse (substrate)',
    'fwd_vs_rev_product': 'Forward vs reverse (product)',
}


def _nan_safe(value, digits=3):
    """
    Format a float to fixed digits, rendering ``nan`` as an en-dash.

    Parameters
    ----------
    value : float
        The value to format.
    digits : int, optional
        Decimal places (default: 3).

    Returns
    -------
    str
        The formatted number, or ``'&ndash;'`` when ``value`` is ``nan``.
    """
    return '&ndash;' if value != value else f'{value:.{digits}f}'


def _flank_comparison_table_html(comparisons):
    """
    Render the five flank-context comparisons as a small HTML table.

    Leads with the scale-free cosine similarity and Cramér's V effect sizes; the
    chi-squared p-value is shown only when both spectra reached the minimum site
    count (``chi2_reliable``), otherwise an en-dash, so sparse per-sequence counts
    are not over-interpreted. A ``*`` marks a reliable p-value below 0.05.

    Parameters
    ----------
    comparisons : dict of str to dict
        The output of
        :func:`derip2.stats.flank_spectra.compare_flank_spectra` (or its pooled
        sibling), keyed by comparison name.

    Returns
    -------
    str
        The ``<table>`` element plus an explanatory caption paragraph.
    """
    import math

    from derip2.stats.flank_spectra import COMPARISON_KEYS

    rows = []
    for key in COMPARISON_KEYS:
        comp = comparisons[key]
        name = _FLANK_COMPARISON_NAMES[key]
        cosine = _nan_safe(comp['cosine_similarity'])
        cramers = _nan_safe(comp['cramers_v'])
        if comp['chi2_reliable']:
            p = comp['pvalue']
            if math.isnan(p):
                p_txt = '&ndash;'
            else:
                star = ' *' if p < 0.05 else ''
                p_txt = ('&lt;0.001' if p < 0.001 else f'{p:.3f}') + star
        else:
            p_txt = '&ndash;'
        n_txt = f'{comp["n_a"]:.0f} / {comp["n_b"]:.0f}'
        rows.append(
            f'<tr><td>{name}</td><td>{cosine}</td><td>{cramers}</td>'
            f'<td>{p_txt}</td><td>{n_txt}</td></tr>'
        )
    body = ''.join(rows)
    return (
        '<table class="flank-compare">'
        '<thead><tr><th>Comparison</th><th>Cosine</th><th>Cram&eacute;r&rsquo;s V</th>'
        '<th>&chi;&sup2; p</th><th>n (a / b)</th></tr></thead>'
        f'<tbody>{body}</tbody></table>'
        '<p class="note">Cosine similarity (1 = identical flank preference) is the '
        'primary effect size; the &chi;&sup2; p-value is shown only where both '
        'spectra have enough sites (otherwise &ndash;), and <code>*</code> marks '
        'p &lt; 0.05.</p>'
    )


def _flank_skipped_note(flank):
    """
    Render the alignment-wide count of sites dropped for an unresolved flank.

    Parameters
    ----------
    flank : derip2.stats.flank_spectra.FlankSpectraResult
        The computed flank spectra.

    Returns
    -------
    str
        A ``<p class="note">`` summarising the per-state skipped counts, or an
        empty string when nothing was skipped.
    """
    total = sum(flank.n_skipped_flank.values())
    if total == 0:
        return ''
    parts = ', '.join(f'{state} {n}' for state, n in flank.n_skipped_flank.items())
    return (
        f'<p class="note">{total} site(s) were skipped for lacking a resolvable '
        f'4&nbsp;bp flank context at an alignment edge ({parts}).</p>'
    )


def _overview_flank_svg(flank):
    """
    Render the pooled flank-context spectra grid for the overview page.

    Parameters
    ----------
    flank : derip2.stats.flank_spectra.FlankSpectraResult
        The computed flank spectra (pooled across all sequences here).

    Returns
    -------
    str
        The inline ``<svg>`` fragment (id-prefixed ``ovwflank-``).
    """
    import matplotlib.pyplot as plt

    from derip2.plotting.flank_spectra import plot_flank_bihistograms_pooled

    fig = plot_flank_bihistograms_pooled(flank, width=11.0, bare=True)
    svg = _figure_to_svg(fig, 'ovwflank-', tight=True)
    plt.close(fig)
    return svg


def _overview_html(
    derip,
    cds_tracks,
    fasta_data=None,
    downloads=(),
    df=None,
    row_to_panel=None,
    spectra_ref_ancestor=None,
    spectra_ref_label=None,
    flank=None,
):
    """
    Build the report's front (overview) page: the full alignment + consensus.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        The analysed DeRIP object.
    cds_tracks : list or None
        Rich per-gene CDS tracks for the alignment figure (``--gff``), or None.
    fasta_data : dict, optional
        FASTA popup payloads (see :func:`_overview_svg`); enables click-to-view.
    downloads : sequence of tuple, optional
        ``(label, filename, text)`` download buttons rendered above the figure as
        self-contained ``data:`` links.
    df : pandas.DataFrame, optional
        The per-sequence statistics; when given, a sortable all-sequence stats
        table (plus the deRIP consensus row) is added below the alignment figure.
    row_to_panel : dict of int to int, optional
        Row-index to panel-position map so the stats table can link each sequence
        name to its per-sequence page (see :func:`_overview_stats_table_html`).
    spectra_ref_ancestor : str or None, optional
        Reference sequence for the pooled spectrum (one base per column). ``None``
        uses the deRIP consensus.
    spectra_ref_label : str or None, optional
        The reference sequence's id, used in the spectrum prose. ``None`` = the
        deRIP-corrected consensus.
    flank : derip2.stats.flank_spectra.FlankSpectraResult or None, optional
        The flank-context spectra; when given, the pooled aggregate flank section
        is added below the mutation spectrum.

    Returns
    -------
    str
        The overview ``<section class="seq-panel">`` (first page of the deck).
    """
    svg = _overview_svg(derip, cds_tracks, fasta_data)
    spectrum_svg = _overview_spectrum_svg(derip, spectra_ref_ancestor)
    n_total = len(derip.alignment)
    n_cols = derip.alignment.get_alignment_length()

    # Download links (self-contained data: URIs) plus a button to view the deRIP
    # sequence FASTA in the popup (the consensus row is also clickable).
    tools = []
    for label, filename, text in downloads:
        tools.append(
            f'<a class="psr-btn" download="{escape(filename, quote=True)}" '
            f'href="{_data_uri(text)}">{escape(label)}</a>'
        )
    if fasta_data and '__derip__' in fasta_data:
        tools.append(
            '<button class="psr-btn" type="button" data-fasta="__derip__">'
            'View deRIP FASTA</button>'
        )
    toolbar = f'<div class="psr-toolbar">{"".join(tools)}</div>' if tools else ''

    spectrum_section = (
        '<h3>Mutation spectrum</h3>'
        '<p class="desc">SBS-96 trinucleotide spectrum of every substitution '
        f'across all sequences relative to {_reference_phrase(spectra_ref_label)}. '
        'The C&rarr;T / G&rarr;A dominance is the RIP signature.</p>'
        f'<div class="spectrum-scroll">{spectrum_svg}</div>'
    )

    # Pooled flank-context section: the alignment-wide 2x3 grid plus the same five
    # substrate-vs-product / strand comparisons run on the pooled counts.
    flank_section = ''
    if flank is not None:
        from derip2.stats.flank_spectra import compare_flank_spectra_pooled

        flank_svg = _overview_flank_svg(flank)
        pooled_cmp = compare_flank_spectra_pooled(flank)
        flank_section = (
            '<h3>Flanking-context spectra of RIP-like sites</h3>'
            '<p class="desc">Pooled across all sequences, as three bihistograms '
            '(surviving <b>substrate</b> CpA/TpG left, realised <b>product</b> TpA '
            'right; CA-state motif on the left axis, equivalent TA-state motif on '
            'the right; reverse-strand motifs folded onto the CpA/TpA strand). A '
            'motif is marked <span style="color:#e34948">*</span> when its '
            'substrate-vs-product enrichment is significant (adjusted standardised '
            'residual, |z|&nbsp;&ge;&nbsp;1.96) &mdash; evidence that local context '
            'influences which substrates escape RIP. At this pooled scale the site '
            'counts are large enough that almost every context is flagged, so read '
            'the effect sizes in the table rather than the marks.</p>'
            f'<div class="spectrum-scroll">{flank_svg}</div>'
            f'{_flank_skipped_note(flank)}'
            f'{_flank_comparison_table_html(pooled_cmp)}'
        )

    stats_section = ''
    if df is not None:
        stats_section = (
            '<h3>Summary statistics</h3>'
            '<p class="desc">Per-sequence statistics for every sequence plus the '
            'deRIP-corrected consensus (its RIP-event and strand-bias columns are '
            'not applicable and shown as &ndash;). Click a sequence name to open '
            'its page, or a column heading to sort.</p>'
            f'{_overview_stats_table_html(df, derip, row_to_panel)}'
        )

    return (
        '<section class="seq-panel" data-index="overview" hidden>'
        f'<h2>Overview <span class="seqlen">({n_total} sequences &times; '
        f'{n_cols} columns)</span>{_zoom_control()}</h2>'
        '<h3>Full alignment</h3>'
        '<p class="desc">The whole alignment with RIP markup and, beneath it, the '
        'deRIP-corrected consensus with corrected positions; any gene-annotation '
        'track is drawn below the consensus. Click the deRIP sequence or a CDS '
        'annotation to view its FASTA.</p>'
        f'{toolbar}'
        f'<div class="aln-scroll">{svg}</div>'
        f'{spectrum_section}'
        f'{flank_section}'
        f'{stats_section}'
        '</section>'
    )


def _fix_wide_axes(fig, width_in, n_cols, *, top_in, bottom_in):
    """
    Pin a wide figure's axes to fixed absolute margins and a common x-range.

    Using absolute (inch) margins converted to fractions keeps the plotting area
    at the same pixel offset regardless of the figure's total width, so the
    alignment-row and strand-bias strips align column-for-column and a scroll
    offset points at the same column on every page.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to adjust (single axes).
    width_in : float
        The figure width in inches.
    n_cols : int
        Number of alignment columns, used to set a shared x-range.
    top_in, bottom_in : float
        Top and bottom margins in inches.

    Returns
    -------
    None
        The axes are repositioned in place.
    """
    left_in, right_in = 0.8, 0.2
    height_in = fig.get_size_inches()[1]
    fig.subplots_adjust(
        left=left_in / width_in,
        right=1.0 - right_in / width_in,
        top=1.0 - top_in / height_in,
        bottom=bottom_in / height_in,
    )
    # A shared x-range so every strip (and any annotation sub-plot) maps columns
    # to identical pixels.
    for axis in fig.axes:
        axis.set_xlim(-0.5, n_cols - 0.5)


def _fix_spectrum_axes(fig):
    """
    Pin the SBS-96 spectrum to a fixed axes rectangle with a padded y-label.

    ``tight_layout`` sizes the left margin to the y tick labels, which vary with
    the count magnitude (``5`` vs ``5000``), shifting the plot body between
    sequences. Fixing the axes rectangle — with enough left margin for the
    widest labels and extra spacing between the axis numbers and the y-axis
    label — keeps the spectrum aligned across pages.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The single-panel spectrum figure.

    Returns
    -------
    None
        The axes are repositioned in place.
    """
    ax = fig.axes[0]
    ax.yaxis.labelpad = 12
    # Fixed rect sized for the wide (96-tick) spectrum; identical on every page.
    ax.set_position([0.09, 0.30, 0.88, 0.46])


def write_per_sequence_report(
    derip,
    output_file,
    *,
    title=None,
    ambiguous='split',
    max_seqs=None,
    gff=None,
    genetic_code=1,
    spectra_ref_index=None,
):
    """
    Write a single-file, arrow-key-navigable per-sequence HTML report.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        A DeRIP object on which ``calculate_rip()`` has already been run.
    output_file : str
        Destination path.
    title : str, optional
        Report heading. Defaults to ``'deRIP2 per-sequence report'``.
    ambiguous : {'split', 'exclude', 'weight', 'both'}, optional
        Ambiguity policy used for the per-sequence RSI statistics
        (default: ``'split'``).
    max_seqs : int, optional
        Cap the number of sequence panels. When the alignment has more sequences
        than this, the strongest strand-bias sequences (largest ``|RSI|``) are
        kept and a truncation note is shown. ``None`` (default) renders every
        sequence.
    gff : str, optional
        Path to a GFF3 gene model. When given, each annotated sequence's panel
        gains a gene-effect table and the deRIP-restored protein.
    genetic_code : int, optional
        NCBI translation table for the effect prediction (default: 1).
    spectra_ref_index : int, optional
        Alignment row index of a sequence to use as the reference for the
        mutation spectra (per-sequence and the pooled overview), instead of the
        default deRIP-corrected consensus. Supports negative indexing. The
        reference sequence's own panel then shows an empty (self-comparison)
        spectrum.

    Returns
    -------
    str
        The path written.

    Raises
    ------
    ValueError
        If ``spectra_ref_index`` is out of range for the alignment.

    Notes
    -----
    Rendering hundreds of sequences produces hundreds of inline-SVG figures and
    a correspondingly large file; ``max_seqs`` is the recommended mitigation for
    large alignments.
    """
    # This whole routine can take a while on large alignments (one panel of ~6
    # figures per sequence, plus the per-row spectra and the overview figure), so
    # each expensive stage announces itself and the per-panel loop shows a bar.
    n_seqs = len(derip.alignment)
    logger.info(f'Building per-sequence report for {n_seqs} sequences...')

    df = derip.summarize_stats(ambiguous=ambiguous)

    # Resolve the mutation-spectra reference. By default every sequence is
    # compared to the deRIP-corrected consensus; a user may instead pick an
    # alignment row (its gapped sequence, one base per column) as the reference.
    spectra_ref_ancestor = None
    spectra_ref_label = None
    if spectra_ref_index is not None:
        if not -n_seqs <= spectra_ref_index < n_seqs:
            raise ValueError(
                f'spectra_ref_index {spectra_ref_index} is out of range for an '
                f'alignment of {n_seqs} sequences (valid: '
                f'{-n_seqs}..{n_seqs - 1}).'
            )
        # Normalise a negative index so row_index comparisons in the panels match.
        spectra_ref_index %= n_seqs
        ref_record = derip.alignment[spectra_ref_index]
        spectra_ref_ancestor = str(ref_record.seq)
        spectra_ref_label = ref_record.id
        logger.info(
            f'Computing mutation spectra against reference row '
            f'{spectra_ref_index} ({spectra_ref_label}) instead of the deRIP '
            f'consensus...'
        )
    else:
        logger.info('Computing per-sequence mutation spectra...')

    # One sample column per sequence, measured against the chosen reference (the
    # reconstructed ancestor by default), in each context. Computed once and
    # reused across every panel.
    spectra = derip.calculate_spectra(partition_by='row', ancestor=spectra_ref_ancestor)
    downstream = derip.calculate_spectra(
        partition_by='row', ancestor=spectra_ref_ancestor, context='downstream'
    )
    # Flank-context spectra of RIP-like sites: one sample column per sequence,
    # always measured against this sequence's own bases (independent of the
    # spectra reference), so computed once here and reused across every panel.
    logger.info('Computing per-sequence flanking-context spectra of RIP-like sites...')
    flank = derip.calculate_flank_spectra()

    # FASTA payloads for the overview downloads + click-to-view popups. The deRIP
    # sequence is always available; CDS records are added when a GFF is supplied.
    # Keyed for the popup JS: '__derip__' for the corrected consensus, then one
    # entry per CDS id. Each value carries the record name, its nucleotide FASTA,
    # its translation FASTA (CDS only) and the genetic-code table used.
    derip_name = derip.consensus.id
    derip_seq = derip.get_consensus_string()
    derip_fasta = _fasta_record(derip_name, derip_seq)
    fasta_data = {
        '__derip__': {
            'name': derip_name,
            'nt': derip_fasta,
            'aa': None,
            'table': None,
        }
    }
    cds_multifasta = None

    # Optional gene-effect data. Parsed once and shared across panels.
    genes_by_seqid = {}
    effects_by_seq = {}
    deripd_aa = {}
    cds_gene_cols = []  # (gene, cds_columns) projected onto shared alignment columns
    overview_track = None  # annotation-track spans for the overview --plot figure
    if gff is not None:
        import numpy as np

        from derip2.annotation import (
            DEFAULT_ANNOTATION_COLORS,
            _read_coding_bases,
            cds_alignment_columns,
            cds_display_id,
            cds_exon_spans,
            cds_stop_columns,
            compute_effects_for_alignment,
            deripd_translations,
            parse_gff3,
            ungapped_to_column_map,
            warn_unmatched_seqids,
        )

        logger.info(f'Computing gene effects from {gff}...')
        genes_by_seqid = parse_gff3(gff)
        warn_unmatched_seqids(genes_by_seqid, [rec.id for rec in derip.alignment])
        effects_by_seq = compute_effects_for_alignment(
            derip, genes_by_seqid, genetic_code=genetic_code
        )
        deripd_aa = deripd_translations(
            derip, genes_by_seqid, genetic_code=genetic_code
        )

        # Project each gene's CDS onto its owning sequence's alignment columns
        # ONCE; the same columns then drive every subject's track (each subject's
        # stop codons are computed per panel).
        id_to_row = {rec.id: i for i, rec in enumerate(derip.alignment)}
        cds_colour = DEFAULT_ANNOTATION_COLORS['CDS']
        for seqid, genes in genes_by_seqid.items():
            ri = id_to_row.get(seqid)
            if ri is None:
                continue
            u2c = ungapped_to_column_map(derip.column_classes.arr[ri])
            for gene in genes:
                cols = cds_alignment_columns(gene, u2c)
                if cols:
                    cds_gene_cols.append(
                        (
                            gene,
                            np.asarray(cols, dtype=int),
                            cds_exon_spans(gene, u2c),
                            cds_colour,
                        )
                    )
        # Rich CDS tracks for the overview --plot figure: stop codons are read
        # off the deRIP'd consensus so the track flags stops in the corrected
        # reading frame (mirrors the per-sequence strips, minus the labels).
        consensus_row = np.frombuffer(
            str(derip.gapped_consensus.seq).upper().encode('ascii'), dtype='S1'
        )
        overview_track = [
            (
                exon_spans,
                gene.strand,
                cds_stop_columns(gene, consensus_row, cols, genetic_code=genetic_code),
                gene.gene_id,
                colour,
                cds_display_id(gene),
            )
            for gene, cols, exon_spans, colour in cds_gene_cols
        ]

        # Per-CDS FASTA payloads, projected onto the deRIP consensus: the coding
        # nucleotides (gaps dropped, minus strand complemented) and the
        # deRIP-restored protein (keyed by parent transcript). Keyed by CDS id for
        # the popup and concatenated into a downloadable multi-FASTA.
        cds_records = []
        for gene, cols, _exon_spans, _colour in cds_gene_cols:
            cds_id = cds_display_id(gene)
            nt, _kept = _read_coding_bases(consensus_row, cols, gene.strand)
            aa = deripd_aa.get(gene.gene_id, '')
            fasta_data[cds_id] = {
                'name': cds_id,
                'nt': _fasta_record(cds_id, nt),
                'aa': _fasta_record(cds_id, aa) if aa else None,
                'table': genetic_code,
            }
            cds_records.append(_fasta_record(cds_id, nt))
        cds_multifasta = ''.join(cds_records) if cds_records else None

    # Overview download buttons: the deRIP sequence, and (with a GFF) every CDS
    # nucleotide sequence as mapped onto the deRIP consensus.
    downloads = [('⭳ deRIP sequence (FASTA)', f'{derip_name}.fasta', derip_fasta)]
    if cds_multifasta:
        downloads.append(('⭳ CDS features (FASTA)', 'deRIP_cds.fasta', cds_multifasta))

    indices, truncated = _select_rows(df, max_seqs)

    # Per-sequence flank-context comparisons, computed only for the rendered rows.
    from derip2.stats.flank_spectra import compare_flank_spectra

    flank_cmp = {row: compare_flank_spectra(flank, row) for row in indices}

    # Map each rendered sequence's alignment-row index to its 1-based panel
    # position (panel 0 is the overview), so the overview stats table can link a
    # sequence name to its page. Sequences dropped by --max-report-seqs are absent
    # and left unlinked.
    row_to_panel = {row_index: pos for pos, row_index in enumerate(indices, start=1)}

    # The front (overview) page — the full alignment + deRIP consensus — followed
    # by one panel per sequence. The overview renders the whole-alignment figure
    # (slow on many rows/columns), so it gets its own message.
    logger.info('Rendering overview page (full alignment + summary)...')
    panels = [
        _overview_html(
            derip,
            overview_track,
            fasta_data,
            downloads,
            df,
            row_to_panel,
            spectra_ref_ancestor,
            spectra_ref_label,
            flank,
        )
    ]
    # The per-sequence panels dominate the runtime on large alignments (each is
    # ~6 matplotlib figures rendered to inline SVG), so show a progress bar.
    logger.info(f'Rendering {len(indices)} sequence panels...')
    panels += [
        _panel_html(
            derip,
            df,
            spectra,
            downstream,
            flank,
            flank_cmp[row_index],
            row_index,
            panel_number,
            effects_by_seq,
            genes_by_seqid,
            deripd_aa,
            cds_gene_cols,
            genetic_code,
            spectra_ref_index,
            spectra_ref_label,
        )
        for panel_number, row_index in tqdm(
            enumerate(indices, start=1),
            total=len(indices),
            desc='Rendering sequence panels',
            unit='seq',
            ncols=80,
            leave=False,
        )
    ]
    logger.info('Assembling HTML report...')

    heading = escape(title or 'deRIP2 per-sequence report')
    n_total = len(df)
    n_shown = len(indices)

    truncation_note = ''
    if truncated:
        truncation_note = (
            f'<p class="note">Showing the {n_shown} sequences with the strongest '
            f'strand bias of {n_total} total (capped by <code>max_seqs</code>). '
            f'Raise <code>--max-report-seqs</code> to include more.</p>'
        )

    nav = (
        '<div class="seq-nav">'
        '<button id="seq-prev" type="button">&larr; Prev</button>'
        '<button id="seq-next" type="button">Next &rarr;</button>'
        '<span class="indicator" id="seq-indicator">Overview</span>'
        '<span class="hint">&larr;/&rarr; keys change page</span>'
        '</div>'
    )

    # FASTA popup payloads, embedded as JSON for the click-to-view modal. The
    # ``</`` escape keeps a sequence/name from prematurely closing the <script>.
    fasta_json = json.dumps(fasta_data).replace('</', '<\\/')

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{heading}</title><style>{_STYLE}{_PSR_STYLE}</style></head>'
        '<body><main>'
        f'<h1>{heading}</h1>'
        f'<p class="sub">{n_total} sequences &times; '
        f'{derip.alignment.get_alignment_length()} columns</p>'
        f'{truncation_note}'
        f'{nav}'
        + ''.join(panels)
        + '<footer>Generated by deRIP2. Figures are inline SVG and render on '
        'their own light surface so the colourblind-safe palette holds.</footer>'
        '<div class="psr-tip" id="psr-tip" hidden></div>'
        + _MODAL_HTML
        + f'<script type="application/json" id="psr-fasta-data">{fasta_json}</script>'
        + f'<script>{_PSR_SCRIPT}</script>'
        '</main></body></html>'
    )

    with open(output_file, 'w', encoding='utf-8') as handle:
        handle.write(html)

    logger.info(f'Per-sequence HTML report written to {output_file}')
    return output_file
