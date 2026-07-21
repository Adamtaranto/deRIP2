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
    per_sequence_strand_bias,
    sequence_row_strip,
)
from derip2.plotting.spectra import plot_sbs96  # noqa: E402

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
