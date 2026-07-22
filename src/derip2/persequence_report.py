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

from html import escape
import logging

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


def _inject_svg_tooltips(svg, prefix, titles):
    """
    Tag named ``<g>`` groups with a ``data-tip`` attribute for custom tooltips.

    Matplotlib writes an artist's ``gid`` as ``<g id="gid">``; ``_figure_to_svg``
    then namespaces every id with ``prefix``. Adding a ``data-tip`` attribute to
    that group's opening tag lets the report's own JavaScript show a floating
    tooltip with no delay and pin it on click (a native ``<title>`` would impose
    a ~1 s browser delay and never appear on click). ``aria-label`` mirrors the
    text so assistive technology can still announce it.

    Parameters
    ----------
    svg : str
        The inline SVG fragment (already id-prefixed).
    prefix : str
        The id prefix applied by :func:`derip2.report._figure_to_svg`.
    titles : dict of str to str
        Maps each artist ``gid`` to its tooltip text.

    Returns
    -------
    str
        The SVG with ``data-tip``/``aria-label`` attributes added.
    """
    for gid, text in titles.items():
        opening = f'<g id="{prefix}{gid}">'
        esc = escape(text, quote=True)
        replacement = f'<g id="{prefix}{gid}" data-tip="{esc}" aria-label="{esc}">'
        svg = svg.replace(opening, replacement, 1)
    return svg


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

/* Significant strand-asymmetry p-value. */
.stat-card td.value.sig { font-weight: 700; }

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
"""

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
      var g = e.target.closest && e.target.closest('[data-tip]');
      if (g) {
        pinned = true; showTip(g.getAttribute('data-tip'), e);
      } else {
        pinned = false; tip.setAttribute('hidden', '');
      }
    });
  }

  show(0);
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
    row_index,
    panel_number,
    effects_by_seq,
    genes_by_seqid,
    deripd_aa,
    cds_gene_cols=(),
    genetic_code=1,
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
        from derip2.annotation import cds_stop_columns

        for gene, cols, exon_spans, colour in cds_gene_cols:
            stops = cds_stop_columns(
                gene, cls.arr[row_index], cols, genetic_code=genetic_code
            )
            cds_tracks.append((exon_spans, gene.strand, stops, gene.gene_id, colour))

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
        '<p class="desc">The single-base-substitution spectrum of this sequence '
        'measured against the reconstructed ancestor, in trinucleotide context '
        '(5′-N[R&gt;A]N-3′). RIP shows up as a C&gt;T peak in CpA context. Scroll '
        'horizontally if the 96 channels do not fit.</p>'
        f'<div class="spectrum-scroll">{sbs_svg}</div>'
        '<h3>Mutation spectrum (downstream context)</h3>'
        '<p class="desc">The same substitutions classified by the mutated base '
        'plus its two downstream bases (pyrimidine-folded), which resolves the '
        'CHG-methylation signal C&gt;T in CpNpG context.</p>'
        f'<div class="spectrum-scroll">{ds_svg}</div>'
        '<h3>Summary statistics</h3>'
        f'{_stats_sections_html(row_stats)}'
        f'{effect_html}'
        f'</section>'
    )


def _overview_svg(derip, cds_tracks):
    """
    Render the full ``--plot`` alignment figure and return it as inline SVG.

    Reuses :meth:`derip2.derip.DeRIP.plot_alignment` (the same figure the
    ``--plot`` flag writes, including the deRIP corrected consensus row). The
    figure is rendered to inline SVG so the axes, annotation track and consensus
    stay crisp vector (the coloured base grid remains a single embedded raster);
    the annotation groups carry ``data-tip`` attributes for the report tooltips.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        The analysed DeRIP object.
    cds_tracks : list or None
        Rich per-gene CDS tracks ``(exon_spans, strand, stop_columns, label,
        colour)`` to draw above the consensus (``--gff``), or None.

    Returns
    -------
    str
        The inline ``<svg>`` fragment (id-prefixed, tooltip-tagged).
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
        svg = _figure_to_svg(fig, 'ovw-', tight=True)
        svg = _inject_svg_tooltips(svg, 'ovw-', getattr(fig, 'annotation_titles', {}))
    finally:
        plt.close(fig)
    return svg


def _overview_html(derip, cds_tracks):
    """
    Build the report's front (overview) page: the full alignment + consensus.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        The analysed DeRIP object.
    cds_tracks : list or None
        Rich per-gene CDS tracks for the alignment figure (``--gff``), or None.

    Returns
    -------
    str
        The overview ``<section class="seq-panel">`` (first page of the deck).
    """
    svg = _overview_svg(derip, cds_tracks)
    n_total = len(derip.alignment)
    n_cols = derip.alignment.get_alignment_length()
    return (
        '<section class="seq-panel" data-index="overview" hidden>'
        f'<h2>Overview <span class="seqlen">({n_total} sequences &times; '
        f'{n_cols} columns)</span>{_zoom_control()}</h2>'
        '<h3>Full alignment</h3>'
        '<p class="desc">The whole alignment with RIP markup and, beneath it, the '
        'deRIP-corrected consensus with corrected positions.</p>'
        f'<div class="aln-scroll">{svg}</div>'
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

    Returns
    -------
    str
        The path written.

    Notes
    -----
    Rendering hundreds of sequences produces hundreds of inline-SVG figures and
    a correspondingly large file; ``max_seqs`` is the recommended mitigation for
    large alignments.
    """
    df = derip.summarize_stats(ambiguous=ambiguous)
    # One sample column per sequence, measured against the reconstructed
    # ancestor, in each context. Computed once and reused across every panel.
    spectra = derip.calculate_spectra(partition_by='row')
    downstream = derip.calculate_spectra(partition_by='row', context='downstream')

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
            cds_alignment_columns,
            cds_exon_spans,
            cds_stop_columns,
            compute_effects_for_alignment,
            deripd_translations,
            parse_gff3,
            ungapped_to_column_map,
            warn_unmatched_seqids,
        )

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
            )
            for gene, cols, exon_spans, colour in cds_gene_cols
        ]

    indices, truncated = _select_rows(df, max_seqs)

    # The front (overview) page — the full alignment + deRIP consensus — followed
    # by one panel per sequence.
    panels = [_overview_html(derip, overview_track)]
    panels += [
        _panel_html(
            derip,
            df,
            spectra,
            downstream,
            row_index,
            panel_number,
            effects_by_seq,
            genes_by_seqid,
            deripd_aa,
            cds_gene_cols,
            genetic_code,
        )
        for panel_number, row_index in enumerate(indices, start=1)
    ]

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
        f'<script>{_PSR_SCRIPT}</script>'
        '</main></body></html>'
    )

    with open(output_file, 'w', encoding='utf-8') as handle:
        handle.write(html)

    logger.info(f'Per-sequence HTML report written to {output_file}')
    return output_file
