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

from derip2.maxrip import (
    MAX_RIP_DESCRIPTIONS,
    MAX_RIP_VARIANTS,
    max_rip_multifasta,
)
from derip2.plotting.persequence import (
    NONRIP_COLOR,
    PRODUCT_COLOR,
    SUBSTRATE_COLOR,
)
from derip2.plotting.strandbias import BASE_COLORS
from derip2.reporting import (
    base_style,
    figure_to_svg,
    inject_svg_tooltips,
    persequence_script,
    persequence_style,
    render_page,
    stat_cell,
)

# Module-level aliases kept for the test-suite and downstream imports.
_STYLE = base_style()
_PSR_STYLE = persequence_style()
_PSR_SCRIPT = persequence_script()

logger = logging.getLogger(__name__)

# Grouped, transposed statistics layout: (section title, description, [(column,
# row label), ...]). Each group becomes a small card with a stat/value table.
#
# The order here drives both views: the per-sequence stat cards
# (:func:`_stats_sections_html`) and the overview summary table
# (:func:`_overview_stats_table_html`). It runs from the headline counts to the
# most specialised measure -- event counts, then the composite index, then
# composition, and finally the strand-bias breakdown, which is the widest group
# and the one a reader is least likely to want first.
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
)


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


def _marked_fasta_html(name, seq, marks, css_class, width=60):
    """
    Render a FASTA record as HTML with selected residues wrapped for emphasis.

    The layout is identical to :func:`_fasta_record` — same header, same line
    wrapping — so the popup's plain-text and highlighted views stay
    character-for-character aligned, and the browser's ``textContent`` of the
    rendered HTML is exactly the plain record (which is what the copy button
    reads). Contiguous runs of marked residues collapse into a single ``<span>``
    to keep the markup small on heavily corrected sequences.

    Parameters
    ----------
    name : str
        The record identifier (the header after ``>``).
    seq : str
        The sequence; wrapped to ``width`` characters per line.
    marks : set of int or collections.abc.Container
        Zero-based offsets into ``seq`` to wrap.
    css_class : str
        Class applied to each emphasis ``<span>``.
    width : int, optional
        Line-wrap width (default: 60), matching :func:`_fasta_record`.

    Returns
    -------
    str
        HTML for the record, ending in a newline.
    """
    out = [f'&gt;{escape(name)}\n']
    open_span = False
    for i, base in enumerate(seq):
        # A newline every ``width`` residues, outside any open span so the markup
        # stays well-formed across line breaks.
        if i and i % width == 0:
            if open_span:
                out.append('</span>')
                open_span = False
            out.append('\n')
        marked = i in marks
        if marked and not open_span:
            out.append(f'<span class="{css_class}">')
            open_span = True
        elif open_span and not marked:
            out.append('</span>')
            open_span = False
        out.append(escape(base))
    if open_span:
        out.append('</span>')
    out.append('\n')
    return ''.join(out)


def _corrected_consensus_offsets(derip):
    """
    Map deRIP-corrected alignment columns onto offsets in the ungapped consensus.

    ``DeRIP.corrected_positions`` is keyed by *gapped* alignment column, but the
    popup shows the ungapped consensus, so the indices have to be rebased by
    dropping the gap columns that precede each correction.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        A DeRIP instance on which ``calculate_rip`` has been run.

    Returns
    -------
    set of int
        Zero-based offsets into ``derip.get_consensus_string()``.

    Raises
    ------
    ValueError
        If a corrected column is a gap in the consensus, which would mean the
        correction record and the consensus had fallen out of step.
    """
    corrected_cols = set(derip.corrected_positions)
    offsets = set()
    ungapped = 0
    for col, base in enumerate(str(derip.gapped_consensus.seq)):
        if base == '-':
            if col in corrected_cols:
                raise ValueError(
                    f'deRIP-corrected column {col} is a gap in the consensus'
                )
            continue
        if col in corrected_cols:
            offsets.add(ungapped)
        ungapped += 1
    return offsets


#: Tab labels for the maximum-RIP popup, in the order they are shown. Ordered
#: least to most aggressive so the tabs read as an escalating series.
_MAX_RIP_LABELS = {
    'observed': 'Observed sites only',
    'all': 'All substrate sites',
    'all_plus_nonrip': 'All sites + non-RIP',
}


def _max_rip_payload(derip):
    """
    Build the popup payload for the maximum-RIP counterfactual sequences.

    One tab per variant, each with its converted sites marked in bold red and a
    footer note explaining that variant's rule and how many sites it changed.

    Parameters
    ----------
    derip : derip2.derip.DeRIP
        A DeRIP instance on which ``calculate_rip`` has been run.

    Returns
    -------
    dict
        A payload with ``name`` and ``tabs``, keyed into the popup as
        ``'__maxrip__'``.
    """
    tabs = []
    for variant in MAX_RIP_VARIANTS:
        result = derip.calculate_max_rip(variant)
        tabs.append(
            _fasta_tab(
                variant,
                _MAX_RIP_LABELS[variant],
                f'{derip.consensus.id}_{variant}',
                result.seq,
                marks=set(result.converted_positions.tolist()),
                css_class='psr-mark-rip',
                note=(
                    f'{MAX_RIP_DESCRIPTIONS[variant]} '
                    f'{result.n_converted:,} site'
                    f'{"" if result.n_converted == 1 else "s"} converted '
                    f'({result.n_forward:,} forward, {result.n_reverse:,} reverse), '
                    'shown in bold red. Copied and downloaded text is plain and '
                    'unformatted.'
                ),
            )
        )
    return {'name': f'{derip.consensus.id} — maximum RIP', 'tabs': tabs}


def _fasta_tab(key, label, name, seq, marks=None, css_class=None, note=None):
    """
    Build one tab of a FASTA popup payload.

    Parameters
    ----------
    key : str
        Stable identifier for the tab, used as the JS tab id.
    label : str
        Text shown on the tab button.
    name : str
        Record identifier for the FASTA header.
    seq : str
        The sequence to show.
    marks : set of int, optional
        Offsets into ``seq`` to highlight; when given (and non-empty) an ``html``
        rendering is attached alongside the plain text.
    css_class : str, optional
        Emphasis class for the marked residues; required when ``marks`` is given.
    note : str, optional
        One-line explanation shown in the popup footer while this tab is active.

    Returns
    -------
    dict
        A tab record with ``key``, ``label``, ``text``, ``html`` and ``note``.
        ``text`` is always the plain FASTA record — it is what the copy button
        and the download links use, so it never carries markup.
    """
    return {
        'key': key,
        'label': label,
        'text': _fasta_record(name, seq),
        'html': (_marked_fasta_html(name, seq, marks, css_class) if marks else None),
        'note': note,
    }


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
            # 'RIP_total' is derived (forward + reverse RIP events); the CRI > 1
            # and p < 0.05 flags are applied by the shared formatter.
            value = (
                int(row['RIP_fwd']) + int(row['RIP_rev'])
                if column == 'RIP_total'
                else row[column]
            )
            text, css = stat_cell(column, value)
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


# The click-to-view FASTA modal, injected once per report. Populated and shown by
# the popup handler in ``_PSR_SCRIPT``; ``data-close`` marks the backdrop and the
# × button so a single handler can dismiss it. The tab strip is deliberately empty:
# each payload declares its own tabs (one for a plain deRIP sequence, two for a CDS
# with a translation, three for the maximum-RIP variants) and the JS builds the
# buttons to match, hiding the strip entirely for single-tab payloads.
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
    '<div class="psr-tabs" id="psr-tabs" hidden></div>'
    '<pre class="psr-fasta" id="psr-fasta"></pre>'
    '<div class="psr-modal-foot">'
    '<button class="psr-copy" id="psr-copy" type="button">Copy</button>'
    '<p class="psr-note" id="psr-note" hidden></p>'
    '</div>'
    '</div></div>'
)


def _select_rows(df, max_seqs):
    """
    Choose which sequence rows to render, and whether the report is truncated.

    When ``max_seqs`` caps the report below the alignment size, the **first**
    ``max_seqs`` rows of the alignment are kept. A prefix is predictable: the
    reader can tell which sequences they are getting without cross-referencing a
    statistic, and re-running with a larger cap only adds panels rather than
    swapping them. (An earlier version kept the strongest strand-bias sequences
    instead, which read as an arbitrary subset.) To report a particular subset,
    reorder or filter the alignment first — for example with
    :meth:`derip2.derip.DeRIP.sort_by_rsi` to put the most strand-biased
    sequences at the top.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :meth:`derip2.derip.DeRIP.summarize_stats`, one row per
        sequence in alignment order.
    max_seqs : int or None
        Maximum number of sequences to render. ``None`` renders all. Values
        below 1 are treated as 1, so the report always has a sequence panel.

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
    return list(range(max(1, max_seqs))), True


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


# Bihistogram description, shown only for the 1 bp flank (16-channel) view.
_FLANK_BIHIST_DESC = (
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
)


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
    strip_svg = figure_to_svg(strip, f's{row_index}row-', tight=False)
    strip_svg = inject_svg_tooltips(
        strip_svg, f's{row_index}row-', getattr(strip, 'annotation_titles', {})
    )
    plt.close(strip)

    bias = per_sequence_strand_bias(cls, row_index, seq_id=seq_id, width=wide_w)
    _fix_wide_axes(bias, wide_w, n_cols, top_in=0.5, bottom_in=0.55)
    bias_svg = figure_to_svg(bias, f's{row_index}bias-', tight=False)
    plt.close(bias)

    completion = rip_completion_bar(row_stats)
    completion_svg = figure_to_svg(completion, f's{row_index}rip-', tight=False)
    plt.close(completion)

    gc = gc_content_bar(row_stats)
    gc_svg = figure_to_svg(gc, f's{row_index}gc-', tight=False)
    plt.close(gc)

    # Wide, bare spectra (no redundant sample title / caption); fixed geometry so
    # the plot body aligns across pages, scrolled independently of the columns.
    # A touch narrower than the page so the scroll box can centre them.
    sbs = plot_sbs96(spectra, sample=row_index, width=11.0, bare=True)
    _fix_spectrum_axes(sbs)
    sbs_svg = figure_to_svg(sbs, f's{row_index}sbs-', tight=False)
    plt.close(sbs)

    ds = plot_downstream(downstream, sample=row_index, width=11.0, bare=True)
    _fix_spectrum_axes(ds)
    ds_svg = figure_to_svg(ds, f's{row_index}ds-', tight=False)
    plt.close(ds)

    # The three flank-context bihistograms (substrate left vs product right, one
    # per strand) as one figure, so a single unique id-prefix keeps the SVG glyph
    # ids collision-free.
    from derip2.plotting.flank_spectra import (
        plot_flank_bihistograms,
        plot_flank_conversion_heatmap,
    )

    # The 16-row bihistogram is only legible for the 1 bp flank (16 channels); for
    # wider flanks the heatmap carries the signal and the bihistogram is omitted.
    flank_bihist_html = ''
    if flank.flank_length == 1:
        flank_fig = plot_flank_bihistograms(flank, sample=row_index, bare=True)
        flank_svg = figure_to_svg(flank_fig, f's{row_index}flank-', tight=True)
        plt.close(flank_fig)
        flank_bihist_html = (
            _FLANK_BIHIST_DESC + f'<div class="spectrum-scroll">{flank_svg}</div>'
        )
    # Interaction heatmap of the same data: % of each flank motif converted
    # substrate -> product. Unique id prefix keeps embedded-SVG glyph ids distinct.
    flank_heat_fig = plot_flank_conversion_heatmap(flank, sample=row_index, bare=True)
    flank_heat_svg = figure_to_svg(
        flank_heat_fig, f's{row_index}flankheat-', tight=True
    )
    plt.close(flank_heat_fig)
    flank_data_table = _flank_data_table_html(
        flank.matrix('substrate', 'combined')[:, row_index],
        flank.matrix('product', 'combined')[:, row_index],
        flank.channels_substrate,
    )
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
        f'{flank_bihist_html}'
        '<p class="desc">The same data as an interaction heatmap: for a RIP target '
        'CpA, the percentage of that flank motif converted from the substrate '
        '(CpA) to the product (TpA) state, as a joint function of the base(s) '
        'immediately 5&prime; (rows) and 3&prime; (columns) of the target.</p>'
        f'<div class="spectrum-scroll">{flank_heat_svg}</div>'
        f'{flank_data_table}'
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
        svg = figure_to_svg(fig, 'ovw-', tight=True)
        svg = inject_svg_tooltips(svg, 'ovw-', titles, fasta_keys)
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
            # Green flags, matching the per-sequence cards: a positive-RIP CRI
            # (> 1) and a significant strand-asymmetry p-value (< 0.05).
            text, css = stat_cell(col, values.get(col), sig_class='pos')
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
    # stat_cell renders them as an en-dash (not applicable to the ancestor).
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
    svg = figure_to_svg(fig, 'ovwsbs-', tight=False)
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


def _flank_data_table_html(substrate, product, motifs):
    """
    Render a sortable per-motif counts table for the flank-context section.

    One row per CA-state flank motif with its combined-strand substrate and
    product counts, their total, and the percentage of that total realised as RIP
    product (the product share = ``product / (substrate + product)``), i.e. how
    readily that flank context is converted. The motif column can be sorted by its
    5' (first) or 3' (last) flanking base; the numeric columns sort by value.

    Parameters
    ----------
    substrate, product : numpy.ndarray
        ``(16,)`` combined-strand substrate and product counts, in canonical flank
        order (aligned to ``motifs``).
    motifs : sequence of str
        The 16 CA-state motif labels (e.g. ``GCAG``), aligned to the counts.

    Returns
    -------
    str
        The sortable ``<table class="flank-data">`` element plus a caption.
    """
    rows = []
    for motif, sub, prod in zip(motifs, substrate, product):
        s = float(sub)
        p = float(prod)
        total = s + p
        if total > 0:
            conv = 100.0 * p / total
            conv_txt = f'{conv:.1f}'
            conv_val = f'{conv:.4f}'
            # Stacked bar: the RIP product share (orange, explicit width) against
            # the surviving substrate share (blue, fills the remainder so the two
            # segments always meet exactly), so a fully-converted context reads
            # all-orange.
            bar = (
                f'<span class="ripbar" title="{conv:.1f}% converted to product">'
                f'<i class="prod" style="width:{conv:.3f}%"></i>'
                f'<i class="sub"></i></span>'
            )
        else:
            conv_txt = '&ndash;'
            conv_val = ''  # sorts last
            bar = '<span class="ripbar empty" title="no sites"></span>'
        rows.append(
            f'<tr><td data-first="{motif[0]}" data-last="{motif[3]}">{motif}</td>'
            f'<td data-val="{s:.0f}">{s:.0f}</td>'
            f'<td data-val="{p:.0f}">{p:.0f}</td>'
            f'<td data-val="{total:.0f}">{total:.0f}</td>'
            f'<td data-val="{conv_val}">{conv_txt}</td>'
            f'<td class="ripbar-cell">{bar}</td></tr>'
        )
    body = ''.join(rows)
    return (
        '<table class="flank-data">'
        '<thead><tr>'
        '<th>Motif <span class="motif-sort">'
        '<button type="button" data-motifsort="first" '
        'title="sort by 5&prime; (first) base">5&prime;</button>'
        '<button type="button" data-motifsort="last" '
        'title="sort by 3&prime; (last) base">3&prime;</button>'
        '</span></th>'
        '<th class="sortable-num" title="click to sort">Substrate</th>'
        '<th class="sortable-num" title="click to sort">Product</th>'
        '<th class="sortable-num" title="click to sort">Total</th>'
        '<th class="sortable-num" title="click to sort">% RIP</th>'
        '<th title="RIP product (orange) vs surviving substrate (blue) share">'
        'Conversion</th>'
        '</tr></thead>'
        f'<tbody>{body}</tbody></table>'
        '<p class="note">Counts pool the forward and reverse strands (combined). '
        '&ldquo;% RIP&rdquo; is the product share of the total &mdash; the fraction '
        'of that flank context converted to RIP product, also shown as a stacked '
        'bar (orange = product, blue = surviving substrate). Sort the motif column '
        'by its 5&prime; or 3&prime; flanking base, or any numeric column by '
        'value.</p>'
    )


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
    Render the pooled flank-context bihistogram for the overview page.

    The overview shows only the **combined**-strand bihistogram (the forward and
    reverse panels are kept for the per-sequence pages), as a single, narrower
    panel.

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

    fig = plot_flank_bihistograms_pooled(
        flank, strands=('combined',), width=5.6, bare=True
    )
    svg = figure_to_svg(fig, 'ovwflank-', tight=True)
    plt.close(fig)
    return svg


def _overview_flank_heatmap_svg(flank, id_prefix='ovwflankheat-'):
    """
    Render the pooled flank-context RIP-conversion heatmap for the overview page.

    Parameters
    ----------
    flank : derip2.stats.flank_spectra.FlankSpectraResult
        The computed flank spectra (pooled across all sequences here).
    id_prefix : str, optional
        Namespace prefix for the SVG's internal ids (default ``'ovwflankheat-'``).
        The overview draws two heatmaps (the primary-width map and a fixed 2 bp
        map), so each must use a distinct prefix to avoid id collisions when both
        SVGs are inlined into the same document.

    Returns
    -------
    str
        The inline ``<svg>`` fragment, id-prefixed with ``id_prefix``.
    """
    import matplotlib.pyplot as plt

    from derip2.plotting.flank_spectra import plot_flank_conversion_heatmap

    fig = plot_flank_conversion_heatmap(flank, sample=None, bare=True)
    svg = figure_to_svg(fig, id_prefix, tight=True)
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
    if fasta_data and '__maxrip__' in fasta_data:
        tools.append(
            '<button class="psr-btn" type="button" data-fasta="__maxrip__">'
            'View maximum RIP sequences</button>'
        )
    toolbar = f'<div class="psr-toolbar">{"".join(tools)}</div>' if tools else ''

    spectrum_section = (
        '<h3>Mutation spectrum</h3>'
        '<p class="desc">SBS-96 trinucleotide spectrum of every substitution '
        f'across all sequences relative to {_reference_phrase(spectra_ref_label)}. '
        'The C&rarr;T / G&rarr;A dominance is the RIP signature.</p>'
        f'<div class="spectrum-scroll">{spectrum_svg}</div>'
    )

    # Pooled flank-context section: the alignment-wide combined-strand bihistogram,
    # a per-motif counts/conversion table, and the five substrate-vs-product /
    # strand comparisons run on the pooled counts.
    flank_section = ''
    if flank is not None:
        from derip2.stats.flank_spectra import compare_flank_spectra_pooled

        flank_heat_svg = _overview_flank_heatmap_svg(flank)

        # Always surface the finer 2 bp interaction map on the overview, in
        # addition to the primary-width heatmap above. Computed via the standalone
        # function so the DeRIP object's cached (primary-width) flank result, reused
        # by every per-sequence page, is left untouched. Skipped when the primary
        # width is already 2 bp (the heatmap above is then the same map).
        flank2_heat_html = ''
        if flank.flank_length != 2:
            from derip2.stats.flank_spectra import compute_flank_spectra

            ids = [record.id for record in derip.alignment]
            flank2 = compute_flank_spectra(
                derip.column_classes, sample_names=ids, flank_length=2
            )
            flank2_heat_svg = _overview_flank_heatmap_svg(
                flank2, id_prefix='ovwflankheat2-'
            )
            flank2_heat_html = (
                '<p class="desc">The same conversion resolved to the <b>2 bp</b> '
                'flank context (16&times;16 grid: the two bases 5&prime; of the '
                'target on the rows, the two bases 3&prime; on the columns, each '
                'ordered nearest-base first). This shows whether the single-base '
                'preference above is carried by the base immediately flanking the '
                'RIP target or extends to the second base out; white cells are '
                'motifs absent from the alignment.</p>'
                f'<div class="spectrum-scroll">{flank2_heat_svg}</div>'
            )

        pooled_cmp = compare_flank_spectra_pooled(flank)
        pooled = flank.pooled()
        flank_data_table = _flank_data_table_html(
            pooled['sub_fwd'] + pooled['sub_rev'],
            pooled['prod_fwd'] + pooled['prod_rev'],
            flank.channels_substrate,
        )
        # The 16-row bihistogram is only legible for the 1 bp flank; omit it for
        # wider flanks where the heatmap carries the signal.
        overview_bihist_html = ''
        if flank.flank_length == 1:
            flank_svg = _overview_flank_svg(flank)
            overview_bihist_html = (
                '<p class="desc">Pooled across all sequences, as a combined-strand '
                'bihistogram (surviving <b>substrate</b> CpA/TpG left, realised '
                '<b>product</b> TpA right; CA-state motif on the left axis, '
                'equivalent TA-state motif on the right; reverse-strand motifs '
                'folded onto the CpA/TpA strand). The per-sequence pages '
                'additionally split this into forward and reverse panels. A motif '
                'is marked <span style="color:#e34948">*</span> when its '
                'substrate-vs-product enrichment is significant (adjusted '
                'standardised residual, |z|&nbsp;&ge;&nbsp;1.96) &mdash; evidence '
                'that local context influences which substrates escape RIP. At this '
                'pooled scale the site counts are large enough that almost every '
                'context is flagged, so read the effect sizes in the table rather '
                'than the marks.</p>'
                f'<div class="spectrum-scroll">{flank_svg}</div>'
            )
        flank_section = (
            '<h3>Flanking-context spectra of RIP-like sites</h3>'
            f'{overview_bihist_html}'
            '<p class="desc">The same pooled data as an interaction heatmap: '
            'the percentage of each flank motif converted from substrate (CpA) to '
            'product (TpA), by the base(s) immediately 5&prime; (rows) and '
            '3&prime; (columns) of the RIP target CpA.</p>'
            f'<div class="spectrum-scroll">{flank_heat_svg}</div>'
            f'{flank2_heat_html}'
            f'{_flank_skipped_note(flank)}'
            f'{flank_data_table}'
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
    flank_length=1,
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
        than this, the first ``max_seqs`` rows in alignment order are kept and a
        truncation note is shown; sort or filter the alignment first (e.g.
        :meth:`derip2.derip.DeRIP.sort_by_rsi`) to change which sequences those
        are. ``None`` (default) renders every sequence.
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
    flank_length : int, optional
        Number of flanking bases each side of a RIP-like dinucleotide for the
        flank-context spectra and conversion heatmap (default 1 → 4×4 grid; 2 →
        16×16).

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
    flank = derip.calculate_flank_spectra(flank_length=flank_length)

    # FASTA payloads for the overview downloads + click-to-view popups. The deRIP
    # sequence and the maximum-RIP variants are always available; CDS records are
    # added when a GFF is supplied. Keyed for the popup JS: '__derip__' for the
    # corrected consensus, '__maxrip__' for the counterfactuals, then one entry
    # per CDS id. Each value is a record name plus a list of tabs.
    derip_name = derip.consensus.id
    derip_seq = derip.get_consensus_string()
    derip_fasta = _fasta_record(derip_name, derip_seq)
    corrected_offsets = _corrected_consensus_offsets(derip)
    n_corrected = len(corrected_offsets)
    fasta_data = {
        '__derip__': {
            'name': derip_name,
            'tabs': [
                _fasta_tab(
                    'nt',
                    'Nucleotide',
                    derip_name,
                    derip_seq,
                    marks=corrected_offsets,
                    css_class='psr-mark',
                    note=(
                        f'{n_corrected:,} position'
                        f'{"" if n_corrected == 1 else "s"} restored by deRIP, '
                        'shown in bold green. Copied and downloaded text is plain '
                        'and unformatted.'
                        if n_corrected
                        else 'No positions were corrected in this alignment.'
                    ),
                )
            ],
        },
        '__maxrip__': _max_rip_payload(derip),
    }
    max_rip_fasta = max_rip_multifasta(
        [derip.calculate_max_rip(variant) for variant in MAX_RIP_VARIANTS],
        seq_id=derip_name,
    )
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
            tabs = [_fasta_tab('nt', 'Nucleotide', cds_id, nt)]
            if aa:
                tabs.append(
                    _fasta_tab(
                        'aa',
                        'Translation',
                        cds_id,
                        aa,
                        note=(f'Translation — NCBI genetic code table {genetic_code}.'),
                    )
                )
            fasta_data[cds_id] = {'name': cds_id, 'tabs': tabs}
            cds_records.append(_fasta_record(cds_id, nt))
        cds_multifasta = ''.join(cds_records) if cds_records else None

    # Overview download buttons: the deRIP sequence, and (with a GFF) every CDS
    # nucleotide sequence as mapped onto the deRIP consensus.
    downloads = [
        ('⭳ deRIP sequence (FASTA)', f'{derip_name}.fasta', derip_fasta),
        (
            '⭳ Maximum RIP sequences (FASTA)',
            f'{derip_name}_maxRIP.fasta',
            max_rip_fasta,
        ),
    ]
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

    n_total = len(df)
    n_shown = len(indices)

    truncation_note = ''
    if truncated:
        truncation_note = (
            f'<p class="note">Showing the first {n_shown} sequences of '
            f'{n_total} total (capped by <code>max_seqs</code>). Raise '
            f'<code>--max-report-seqs</code> to include more, or sort the '
            f'alignment first to change which sequences appear.</p>'
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

    html = render_page(
        title or 'deRIP2 per-sequence report',
        f'{n_total} sequences &times; {derip.alignment.get_alignment_length()} columns',
        truncation_note + nav + ''.join(panels),
        css=_STYLE + _PSR_STYLE,
        scripts=(
            '<div class="psr-tip" id="psr-tip" hidden></div>'
            + _MODAL_HTML
            + f'<script type="application/json" id="psr-fasta-data">{fasta_json}</script>'
            + f'<script>{_PSR_SCRIPT}</script>'
        ),
    )

    with open(output_file, 'w', encoding='utf-8') as handle:
        handle.write(html)

    logger.info(f'Per-sequence HTML report written to {output_file}')
    return output_file
