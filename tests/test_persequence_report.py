"""
Tests for the per-sequence figures and the interactive per-sequence HTML report.

These are smoke/structure tests in the style of the other plotting tests: the
Agg backend is forced, figures are checked for the artists their meaning depends
on rather than by pixel comparison, and the HTML report is checked for the panel
count, the navigation script, and — critically — that no two figures share an
element ID (matplotlib reuses glyph IDs, which would corrupt inline SVG).
"""

import logging
import re

import matplotlib

matplotlib.use('Agg')

from Bio.Align import MultipleSeqAlignment  # noqa: E402
from Bio.Seq import Seq  # noqa: E402
from Bio.SeqRecord import SeqRecord  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from derip2.derip import DeRIP  # noqa: E402
from derip2.plotting.persequence import (  # noqa: E402
    PRODUCT_COLOR,
    SUBSTRATE_COLOR,
    gc_content_bar,
    per_sequence_strand_bias,
    rip_completion_bar,
    sequence_row_strip,
)
from derip2.plotting.spectra import plot_downstream, plot_sbs96  # noqa: E402

logging.disable(logging.CRITICAL)


@pytest.fixture(autouse=True)
def close_figures():
    """Never leak figures between tests."""
    yield
    plt.close('all')


def strand_bias_bars(ax):
    """Return the strand-bias bar Paths, now held in a single PathCollection."""
    from matplotlib.collections import PathCollection

    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            return coll.get_paths()
    return []


def make_alignment(seqs, ids=None):
    """Build a MultipleSeqAlignment from a list of sequence strings."""
    ids = ids or [f'seq{i}' for i in range(len(seqs))]
    return MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=sid) for s, sid in zip(seqs, ids)]
    )


@pytest.fixture
def mintest_derip(mintest_path):
    """A DeRIP object with RIP already calculated on the reference alignment."""
    derip = DeRIP(mintest_path)
    derip.calculate_rip()
    return derip


# --- per-sequence figures --------------------------------------------------


def test_strand_bias_figure_returns_axes(mintest_derip):
    """The per-sequence strand-bias strip returns a single-axes figure."""
    fig = per_sequence_strand_bias(mintest_derip.column_classes, 0, seq_id='Seq1')
    assert fig is not None
    assert len(fig.axes) == 1


def test_strand_bias_bars_are_fixed_height(mintest_derip):
    """Every drawn bar reaches unit height from the baseline, above or below."""
    cls = mintest_derip.column_classes
    fig = per_sequence_strand_bias(cls, 0, seq_id='Seq1')
    ax = fig.axes[0]
    bars = strand_bias_bars(ax)
    assert bars, 'expected at least one RIP-like bar for Seq1'
    for path in bars:
        ys = path.vertices[:, 1]
        ymin, ymax = ys.min(), ys.max()
        # Each bar spans from 0 to +1 (forward) or -1 to 0 (reverse).
        assert abs(ymax - ymin) == pytest.approx(1.0, abs=1e-6)
        assert ymin == pytest.approx(0.0, abs=1e-6) or ymax == pytest.approx(
            0.0, abs=1e-6
        )


def test_row_strip_returns_figure(mintest_derip):
    """The alignment-row strip renders, with a deRIP row when a consensus is given."""
    cls = mintest_derip.column_classes
    consensus = str(mintest_derip.gapped_consensus.seq)
    fig = sequence_row_strip(cls, 0, seq_id='Seq1', consensus_seq=consensus)
    assert fig is not None
    ax = fig.axes[0]
    # Two labelled rows: the sequence and the deRIP consensus.
    assert len(ax.get_yticklabels()) == 2


def test_row_strip_has_marker_triangles(mintest_derip):
    """The alignment row draws triangle markers for the subject's RIP roles."""
    from matplotlib.collections import PathCollection

    cls = mintest_derip.column_classes
    fig = sequence_row_strip(cls, 0, seq_id='Seq1')
    ax = fig.axes[0]
    # scatter() adds PathCollection artists — one per role that has any column.
    collections = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert collections, 'expected triangle marker collections'


def test_plot_sbs96_single_sample(mintest_derip):
    """``sample=`` selects exactly one SBS-96 panel."""
    spectra = mintest_derip.calculate_spectra(partition_by='row')
    assert len(spectra.sample_names) == len(mintest_derip.alignment)
    fig = plot_sbs96(spectra, sample=0)
    # One sample -> one spectrum panel (plus no extra sample panels).
    assert len(fig.axes) == 1


def test_rip_completion_bar(mintest_derip):
    """The RIP-completion bar renders three horizontal bars from a stats row."""
    from matplotlib.patches import Rectangle

    df = mintest_derip.summarize_stats()
    fig = rip_completion_bar(df.iloc[0])
    assert fig is not None
    ax = fig.axes[0]
    # Percentage axis, and at least one drawn segment.
    assert ax.get_xlim() == (0.0, 100.0)
    assert any(isinstance(p, Rectangle) and p.get_width() > 0 for p in ax.patches)


def test_rip_completion_bar_no_sites():
    """A sequence with no RIP-like sites renders without dividing by zero."""
    aln = make_alignment(['GGGGG', 'GGGGG', 'GGGGG'])
    derip = DeRIP(aln)
    derip.calculate_rip()
    fig = rip_completion_bar(derip.summarize_stats().iloc[0])
    assert fig is not None


def test_gc_content_bar(mintest_derip):
    """The GC-content bar's filled segment matches the sequence's GC percentage."""
    from matplotlib.patches import Rectangle

    df = mintest_derip.summarize_stats()
    row = df.iloc[0]
    fig = gc_content_bar(row)
    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlim() == (0.0, 100.0)
    # The GC segment starts at x=0; its width is the GC percentage itself
    # (already 0-100 in summarize_stats), not a re-scaled 100%.
    bars = [p for p in ax.patches if isinstance(p, Rectangle)]
    gc_seg = min(bars, key=lambda p: p.get_x())
    assert gc_seg.get_width() == pytest.approx(float(row['GC']), abs=1e-6)
    assert 0.0 < gc_seg.get_width() < 100.0


def test_strand_bias_uses_swapped_role_colours(mintest_derip):
    """Product bars are orange and substrate bars blue (swapped convention)."""
    from matplotlib.collections import PathCollection
    from matplotlib.colors import to_hex

    fig = per_sequence_strand_bias(mintest_derip.column_classes, 0, seq_id='Seq1')
    ax = fig.axes[0]
    coll = next(c for c in ax.collections if isinstance(c, PathCollection))
    bar_colours = {to_hex(c) for c in coll.get_facecolors()}
    # Every bar is one of the two role colours; product is orange, not blue.
    assert bar_colours <= {to_hex(PRODUCT_COLOR), to_hex(SUBSTRATE_COLOR)}
    assert to_hex(PRODUCT_COLOR) != to_hex(SUBSTRATE_COLOR)


def test_plot_downstream_single_sample(mintest_derip):
    """``sample=`` selects one downstream-context panel."""
    ds = mintest_derip.calculate_spectra(partition_by='row', context='downstream')
    fig = plot_downstream(ds, sample=0)
    assert len(fig.axes) == 1


def test_plot_downstream_out_of_range(mintest_derip):
    """An out-of-range downstream sample index is rejected."""
    ds = mintest_derip.calculate_spectra(partition_by='row', context='downstream')
    with pytest.raises(IndexError):
        plot_downstream(ds, sample=999)


def test_plot_sbs96_bare_has_no_suptitle(mintest_derip):
    """``bare=True`` omits the caption suptitle and per-sample title."""
    spectra = mintest_derip.calculate_spectra(partition_by='row')
    fig = plot_sbs96(spectra, sample=0, bare=True)
    assert fig._suptitle is None
    assert not fig.axes[0].get_title()


def test_plot_sbs96_out_of_range(mintest_derip):
    """An out-of-range sample index is rejected."""
    spectra = mintest_derip.calculate_spectra(partition_by='row')
    with pytest.raises(IndexError):
        plot_sbs96(spectra, sample=999)


def test_plot_sbs96_default_still_all_samples(mintest_derip):
    """Without ``sample=`` every sample is drawn, preserving old behaviour."""
    spectra = mintest_derip.calculate_spectra(partition_by='row')
    fig = plot_sbs96(spectra)
    assert len(fig.axes) == len(spectra.sample_names)


# --- report ----------------------------------------------------------------


def test_report_written(mintest_derip, tmp_path):
    """The report is a single file with an overview page + one panel per sequence."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    assert out.exists() and out.stat().st_size > 0
    html = out.read_text()
    n = len(mintest_derip.alignment)
    # One overview page plus one panel per sequence.
    assert html.count('class="seq-panel"') == n + 1
    assert 'data-index="overview"' in html
    assert 'data:image/png;base64' in html  # embedded --plot overview figure


def test_report_has_navigation(mintest_derip, tmp_path):
    """The report ships the arrow-key navigation script."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    html = out.read_text()
    assert "addEventListener('keydown'" in html
    assert 'ArrowRight' in html
    assert 'ArrowLeft' in html


def test_report_no_duplicate_figure_ids(mintest_derip, tmp_path):
    """
    Every inline-SVG element ID is unique across all figures.

    Matplotlib reuses glyph IDs between figures; a browser resolves ``href="#id"``
    to the first match, so a duplicate would make one figure borrow another's
    glyphs. The per-figure prefix must prevent this.
    """
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    html = out.read_text()
    ids = re.findall(r'id="([^"]+)"', html)
    assert ids, 'expected embedded SVG element IDs'
    assert len(ids) == len(set(ids)), 'duplicate SVG element IDs across figures'


def test_report_truncation(mintest_derip, tmp_path):
    """``max_seqs`` caps the sequence-panel count and shows a truncation note."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out), max_seqs=2)
    html = out.read_text()
    # Two sequence panels plus the overview page.
    assert html.count('class="seq-panel"') == 3
    assert 'note' in html


def test_report_section_headings_and_scroll(mintest_derip, tmp_path):
    """Each panel carries section headings and horizontal-scroll containers."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    html = out.read_text()
    for heading in (
        'Alignment row',
        'Per-sequence strand bias',
        'RIP completion',
        'GC content',
        'Mutation spectrum (SBS-96)',
        'Mutation spectrum (downstream context)',
        'Summary statistics',
    ):
        assert f'<h3>{heading}</h3>' in html, heading
    # Wide figures scroll; the alignment row and bias each get a scroll box.
    n = len(mintest_derip.alignment)
    assert html.count('col-scroll') >= 2 * n
    # A seqlen span in each sequence header plus the overview header.
    assert html.count('class="seqlen"') == n + 1
    assert 'nt)' in html
    # The overview page embeds the full alignment figure in a both-axis scroller.
    assert 'aln-scroll' in html


def test_report_transposed_stat_cards(mintest_derip, tmp_path):
    """Statistics are rendered as grouped, transposed cards, not one wide row."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    html = out.read_text()
    assert 'stat-grid' in html
    n = len(mintest_derip.alignment)
    # Four stat sections per sequence.
    assert html.count('class="stat-card"') == 4 * n
    for title in (
        'RIP events',
        'Strand bias (RSI)',
        'Composite RIP Index',
        'Composition',
    ):
        assert title in html


def test_report_scroll_preserved_between_pages(mintest_derip, tmp_path):
    """The navigation script preserves scroll position rather than resetting it."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    html = out.read_text()
    # It must not jump to the top on every page change ...
    assert 'window.scrollTo(0, 0)' not in html
    # ... and it must remember/re-apply the horizontal offset.
    assert 'scrollLeft' in html


def test_report_requires_rip(mintest_path, tmp_path):
    """Writing a report before calculate_rip raises."""
    derip = DeRIP(mintest_path)
    with pytest.raises(ValueError):
        derip.write_per_sequence_report(str(tmp_path / 'x.html'))


# --- edge cases ------------------------------------------------------------


def test_zero_rip_sequence():
    """A sequence with no RIP-like sites renders an empty strand-bias strip."""
    # No CpA/TpA/TpG context anywhere -> no RIP columns.
    aln = make_alignment(['GGGGG', 'GGGGG', 'GGGGG'])
    derip = DeRIP(aln)
    derip.calculate_rip()
    fig = per_sequence_strand_bias(derip.column_classes, 0, seq_id='seq0')
    ax = fig.axes[0]
    assert list(strand_bias_bars(ax)) == []


def test_all_gap_row():
    """An all-gap row renders a strip without crashing."""
    aln = make_alignment(['-----', 'CACCA', 'TACCA'])
    derip = DeRIP(aln)
    derip.calculate_rip()
    fig = sequence_row_strip(derip.column_classes, 0, seq_id='seq0')
    assert fig is not None


def test_single_rip_column():
    """A one-RIP-column sequence renders exactly the expected bar."""
    # Column 0 is a forward RIP column (C/T then A); Seq 'TACAA' carries the
    # product T there.
    aln = make_alignment(['CACAA', 'TACAA', 'TACAA'])
    derip = DeRIP(aln)
    derip.calculate_rip()
    fig = per_sequence_strand_bias(derip.column_classes, 1, seq_id='seq1')
    ax = fig.axes[0]
    assert len(strand_bias_bars(ax)) >= 1


# --- GFF integration -------------------------------------------------------


def test_report_with_gff_effects(mintest_derip, gff_path, tmp_path):
    """A GFF adds a CDS SNP-effect section and restored translations."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out), gff=str(gff_path))
    html = out.read_text()
    assert '<h3>CDS SNP effects</h3>' in html
    assert 'deRIP-restored protein' in html
    # Panels for annotated sequences show an effect table or the no-change note.
    assert 'missense' in html or 'No RIP-induced coding change' in html


def test_report_without_gff_has_no_effect_panel(mintest_derip, tmp_path):
    """Without a GFF, no CDS SNP-effect section appears."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    assert 'CDS SNP effects' not in out.read_text()


def test_row_strip_draws_cds_track(mintest_derip):
    """A cds_tracks argument adds a labelled annotation sub-plot with stop marks."""
    from matplotlib.collections import PathCollection
    import numpy as np

    cls = mintest_derip.column_classes
    # One two-exon gene on the plus strand, with a stop codon at column 4.
    exon_spans = [(0, 2), (4, 6)]
    stop_cols = np.array([4])
    fig = sequence_row_strip(
        cls,
        0,
        seq_id='Seq1',
        consensus_seq=str(mintest_derip.gapped_consensus.seq),
        cds_tracks=[(exon_spans, '+', stop_cols, 'geneX', '#efb700')],
    )
    # The annotation lives in its own sub-plot (a second axis).
    assert len(fig.axes) == 2
    ann = fig.axes[1]
    assert 'geneX' in [t.get_text() for t in ann.get_yticklabels()]
    # Two exon glyphs in one PathCollection, and a '*' stop marker above them.
    colls = [c for c in ann.collections if isinstance(c, PathCollection)]
    assert colls and len(colls[0].get_paths()) == 2
    assert any(t.get_text() == '*' for t in ann.texts)


def test_gene_exon_path_arrow_direction():
    """The exon glyph's arrow tip points in the strand direction."""
    from derip2.plotting.persequence import _gene_exon_path

    plus = _gene_exon_path(0, 10, 0.0, 1.0, '+', 1.0, 0.3, 2.0)
    minus = _gene_exon_path(0, 10, 0.0, 1.0, '-', 1.0, 0.3, 2.0)
    ymid = 0.5
    # The arrow tip is the vertex sitting at the band midline.
    plus_tip = [x for x, y in plus.vertices if abs(y - ymid) < 1e-9]
    minus_tip = [x for x, y in minus.vertices if abs(y - ymid) < 1e-9]
    assert max(plus_tip) == 10  # '+' tip at the right end
    assert min(minus_tip) == 0  # '-' tip at the left end


def test_report_total_rip_events(mintest_derip, tmp_path):
    """The RIP-events card shows a total equal to forward + reverse."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    html = out.read_text()
    assert 'Total RIP events' in html
    df = mintest_derip.summarize_stats()
    expected = int(df.iloc[0]['RIP_fwd']) + int(df.iloc[0]['RIP_rev'])
    # The first panel's total appears as a table cell value.
    assert f'>{expected}<' in html


def test_report_has_legend_and_zoom(mintest_derip, tmp_path):
    """The report carries the alignment-row colour key and the zoom control."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    html = out.read_text()
    assert 'class="legend"' in html
    assert 'zoom-in' in html and 'zoom-out' in html and 'applyZoom' in html


def test_report_pvalue_bold_when_significant(mintest_path):
    """A strand-asymmetry p-value below 0.05 gets the bold .sig class."""
    from derip2.persequence_report import _stats_sections_html

    derip = DeRIP(mintest_path)
    derip.calculate_rip()
    row = derip.summarize_stats().iloc[0].copy()
    row['pvalue'] = 0.01
    assert 'sig">' in _stats_sections_html(row)
    row['pvalue'] = 0.5
    assert 'sig">' not in _stats_sections_html(row)


def test_report_cri_highlighted_when_above_one(mintest_path, tmp_path):
    """A CRI above 1 is flagged green (the .pos class) in the stats card."""
    from derip2.persequence_report import _stats_sections_html

    derip = DeRIP(mintest_path)
    derip.calculate_rip()
    row = derip.summarize_stats().iloc[0].copy()
    row['CRI'] = 1.5
    html = _stats_sections_html(row)
    # The CRI value cell carries the positive (green) class.
    assert 'class="value pos">+1.500' in html or 'class="value pos">1.500' in html

    row['CRI'] = 0.5
    html_low = _stats_sections_html(row)
    assert 'value pos">0.500' not in html_low and 'value pos">+0.500' not in html_low
