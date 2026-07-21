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

from derip2.report import (
    _STYLE,
    _figure_to_svg,
    _stats_table_html,
)

logger = logging.getLogger(__name__)

# Extra CSS layered on top of the shared report style: the panel show/hide
# mechanism and the navigation bar. Kept minimal and theme-agnostic (it inherits
# the light/dark variables from ``_STYLE``).
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
.seq-panel h2 .seqid { color: var(--ink-2); font-weight: 400; }
.note { color: var(--muted); font-size: 13px; }
"""

# ~30 lines of dependency-free navigation: track the visible panel, wrap at the
# ends, and bind the arrow keys plus the prev/next buttons.
_PSR_SCRIPT = """
(function () {
  var panels = Array.prototype.slice.call(
    document.querySelectorAll('.seq-panel'));
  if (!panels.length) return;
  var indicator = document.getElementById('seq-indicator');
  var current = 0;
  function show(k) {
    current = (k + panels.length) % panels.length;
    panels.forEach(function (p, i) {
      if (i === current) { p.removeAttribute('hidden'); }
      else { p.setAttribute('hidden', ''); }
    });
    if (indicator) {
      indicator.textContent = 'Sequence ' + (current + 1) + ' / ' + panels.length;
    }
    window.scrollTo(0, 0);
  }
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


def _panel_html(derip, df, spectra, row_index, panel_number):
    """
    Build the HTML for one sequence's panel.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        The analysed DeRIP object.
    df : pandas.DataFrame
        The per-sequence statistics table.
    spectra : derip2.stats.mutation_spectra.SpectraResult
        Per-row spectra (one sample column per sequence).
    row_index : int
        Alignment row index of the sequence.
    panel_number : int
        1-based position of this panel among those rendered (for the heading).

    Returns
    -------
    str
        The ``<section>`` element for this sequence.
    """
    import matplotlib.pyplot as plt

    from derip2.plotting.persequence import (
        per_sequence_strand_bias,
        sequence_row_strip,
    )
    from derip2.plotting.spectra import plot_sbs96

    cls = derip.column_classes
    seq_id = derip.alignment[row_index].id
    consensus_seq = str(derip.gapped_consensus.seq)

    figures = []

    strip = sequence_row_strip(
        cls, row_index, seq_id=seq_id, consensus_seq=consensus_seq
    )
    figures.append(('Alignment row', _figure_to_svg(strip, f's{row_index}row-')))
    plt.close(strip)

    bias = per_sequence_strand_bias(cls, row_index, seq_id=seq_id)
    figures.append(('Strand bias', _figure_to_svg(bias, f's{row_index}bias-')))
    plt.close(bias)

    sbs = plot_sbs96(spectra, sample=row_index, title=f'SBS-96: {seq_id}')
    figures.append(('Mutation spectrum', _figure_to_svg(sbs, f's{row_index}sbs-')))
    plt.close(sbs)

    figure_html = ''.join(f'<div class="figure">{svg}</div>' for _label, svg in figures)
    stats_html = _stats_table_html(df.iloc[[row_index]])

    return (
        f'<section class="seq-panel" data-index="{panel_number - 1}" hidden>'
        f'<h2>Sequence {panel_number}: <span class="seqid">{escape(seq_id)}</span></h2>'
        f'{figure_html}'
        f'<h2>Summary statistics</h2>'
        f'<div class="table-wrap">{stats_html}</div>'
        f'</section>'
    )


def write_per_sequence_report(
    derip,
    output_file,
    *,
    title=None,
    ambiguous='split',
    max_seqs=None,
    **kwargs,
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
    **kwargs
        Reserved for later use (e.g. GFF gene-effect panels); ignored for now.

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
    # One SBS-96 sample column per sequence, measured against the reconstructed
    # ancestor. Computed once and reused across every panel.
    spectra = derip.calculate_spectra(partition_by='row')

    indices, truncated = _select_rows(df, max_seqs)

    panels = [
        _panel_html(derip, df, spectra, row_index, panel_number)
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
        f'<span class="indicator" id="seq-indicator">Sequence 1 / {n_shown}</span>'
        '<span class="hint">Use the &larr; and &rarr; arrow keys</span>'
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
        f'<script>{_PSR_SCRIPT}</script>'
        '</main></body></html>'
    )

    with open(output_file, 'w', encoding='utf-8') as handle:
        handle.write(html)

    logger.info(f'Per-sequence HTML report written to {output_file}')
    return output_file
