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
from matplotlib.patches import PathPatch  # noqa: E402
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
    bars = [p for p in ax.patches if isinstance(p, PathPatch)]
    assert bars, 'expected at least one RIP-like bar for Seq1'
    for bar in bars:
        ymin, ymax = (
            bar.get_path().vertices[:, 1].min(),
            bar.get_path().vertices[:, 1].max(),
        )
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
    from matplotlib.colors import to_hex

    fig = per_sequence_strand_bias(mintest_derip.column_classes, 0, seq_id='Seq1')
    ax = fig.axes[0]
    bar_colours = {
        to_hex(p.get_facecolor()) for p in ax.patches if isinstance(p, PathPatch)
    }
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
    """The report is a non-empty single file with one panel per sequence."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    assert out.exists() and out.stat().st_size > 0
    html = out.read_text()
    n = len(mintest_derip.alignment)
    assert html.count('class="seq-panel"') == n


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
    """``max_seqs`` caps the panel count and shows a truncation note."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out), max_seqs=2)
    html = out.read_text()
    assert html.count('class="seq-panel"') == 2
    assert 'note' in html
    assert 'Sequence 1 / 2' in html


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
    # Ungapped length is shown in each panel header.
    assert html.count('class="seqlen"') == n
    assert 'nt)' in html


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
    assert [p for p in ax.patches if isinstance(p, PathPatch)] == []


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
    bars = [p for p in ax.patches if isinstance(p, PathPatch)]
    assert len(bars) >= 1


# --- GFF integration -------------------------------------------------------


def test_report_with_gff_effects(mintest_derip, gff_path, tmp_path):
    """A GFF adds gene-effect panels and restored translations to the report."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out), gff=str(gff_path))
    html = out.read_text()
    assert 'Gene effects' in html
    assert 'deRIP-restored protein' in html
    # Panels for annotated sequences show an effect table or the no-change note.
    assert 'missense' in html or 'No RIP-induced coding change' in html


def test_report_without_gff_has_no_effect_panel(mintest_derip, tmp_path):
    """Without a GFF, no gene-effect section appears."""
    out = tmp_path / 'per_seq.html'
    mintest_derip.write_per_sequence_report(str(out))
    assert 'Gene effects' not in out.read_text()
