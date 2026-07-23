"""
Tests for the flanking-context spectra figures.

These render headless (Agg) and assert structural properties — panel count,
bar colours, headings — rather than pixel output.
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import pytest
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from derip2.aln_ops import classify_alignment
from derip2.plotting.flank_spectra import (
    PRODUCT_COLOR,
    SUBSTRATE_COLOR,
    plot_flank_spectra_grid,
    plot_flank_spectra_pooled,
)
from derip2.stats.flank_spectra import compute_flank_spectra


def make_result(seqs):
    """Classify a hand-built alignment and compute its flank spectra."""
    align = MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=f'seq{i}') for i, s in enumerate(seqs)]
    )
    cls = classify_alignment(align, progress=False)
    return compute_flank_spectra(
        cls, sample_names=[f'seq{i}' for i in range(len(seqs))]
    )


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test to bound memory."""
    yield
    plt.close('all')


def test_grid_returns_figure_with_six_axes():
    """The 2x3 grid has exactly six axes."""
    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    fig = plot_flank_spectra_grid(result, sample=0)
    assert len(fig.axes) == 6


def test_grid_uses_state_colours():
    """Substrate row bars are blue, product row bars orange."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_spectra_grid(result, sample=1)
    # Axes are row-major: first three are substrate, last three product.
    sub_ax, prod_ax = fig.axes[0], fig.axes[3]
    sub_colours = {to_hex(p.get_facecolor()) for p in sub_ax.patches}
    prod_colours = {to_hex(p.get_facecolor()) for p in prod_ax.patches}
    assert to_hex(SUBSTRATE_COLOR) in sub_colours
    assert to_hex(PRODUCT_COLOR) in prod_colours


def test_grid_has_row_and_column_labels():
    """Column titles name the strands; first-column y-labels name the states."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_spectra_grid(result, sample=0)
    titles = [ax.get_title() for ax in fig.axes]
    assert 'Combined' in titles and 'Forward' in titles and 'Reverse' in titles
    ylabels = [ax.get_ylabel() for ax in fig.axes]
    assert any('Substrate' in y for y in ylabels)
    assert any('Product' in y for y in ylabels)


def test_grid_bare_has_no_suptitle():
    """bare=True omits the caption/suptitle for embedding."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_spectra_grid(result, sample=0, bare=True)
    assert fig._suptitle is None


def test_grid_sample_out_of_range():
    """An out-of-range sample index raises IndexError."""
    result = make_result(['GCAT', 'GTAT'])
    with pytest.raises(IndexError):
        plot_flank_spectra_grid(result, sample=9)


def test_grid_negative_index_wraps():
    """A negative sample index selects from the end without error."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_spectra_grid(result, sample=-1)
    assert len(fig.axes) == 6


def test_pooled_returns_figure_with_six_axes():
    """The pooled overview grid also has six axes."""
    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    fig = plot_flank_spectra_pooled(result)
    assert len(fig.axes) == 6


def test_grid_saves_to_file(tmp_path):
    """An outfile path writes a figure to disk."""
    result = make_result(['GCAT', 'GTAT'])
    out = tmp_path / 'flank.png'
    plot_flank_spectra_grid(result, sample=0, outfile=str(out))
    assert out.exists() and out.stat().st_size > 0
