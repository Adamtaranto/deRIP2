"""
Tests for the flanking-context spectra bihistogram figures.

These render headless (Agg) and assert structural properties — panel count, bar
direction and colours, CA-state labels, significance marks — rather than pixel
output.
"""

import matplotlib

matplotlib.use('Agg')
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pytest

from derip2.aln_ops import classify_alignment
from derip2.plotting.flank_spectra import (
    PRODUCT_COLOR,
    SIG_COLOR,
    SUBSTRATE_COLOR,
    plot_flank_bihistograms,
    plot_flank_bihistograms_pooled,
)
from derip2.spectra.flank_channels import FLANK16_LABELS_CA, FLANK16_LABELS_TA
from derip2.stats.flank_spectra import FlankSpectraResult, compute_flank_spectra


def make_result(seqs):
    """Classify a hand-built alignment and compute its flank spectra."""
    align = MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=f'seq{i}') for i, s in enumerate(seqs)]
    )
    cls = classify_alignment(align, progress=False)
    return compute_flank_spectra(
        cls, sample_names=[f'seq{i}' for i in range(len(seqs))]
    )


def _bar_rects(ax):
    """The bar Rectangles of an axes (excludes axhspan Polygons)."""
    return [p for p in ax.patches if isinstance(p, Rectangle)]


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure after each test to bound memory."""
    yield
    plt.close('all')


def test_returns_three_panels():
    """The figure has exactly three bihistogram panels (one per strand)."""
    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    fig = plot_flank_bihistograms(result, sample=0)
    assert len(fig.axes) == 3


def _both_states_result():
    """A one-sample result with both substrate and product counts populated."""
    sub_fwd = np.zeros((16, 1))
    prod_fwd = np.zeros((16, 1))
    sub_fwd[3, 0] = 7.0
    prod_fwd[9, 0] = 4.0
    return FlankSpectraResult(
        sub_fwd=sub_fwd,
        sub_rev=np.zeros((16, 1)),
        prod_fwd=prod_fwd,
        prod_rev=np.zeros((16, 1)),
        sample_names=['s'],
        n_skipped_flank=dict.fromkeys(
            ('sub_fwd', 'sub_rev', 'prod_fwd', 'prod_rev'), 0
        ),
    )


def test_substrate_left_product_right():
    """Substrate bars extend left (negative width), product bars right."""
    fig = plot_flank_bihistograms(_both_states_result(), sample=0)
    widths = [r.get_width() for r in _bar_rects(fig.axes[0])]
    assert any(w < 0 for w in widths)  # substrate on the left
    assert any(w > 0 for w in widths)  # product on the right


def test_uses_state_colours():
    """The two states keep the shared blue-substrate / orange-product palette."""
    fig = plot_flank_bihistograms(_both_states_result(), sample=0)
    colours = {to_hex(r.get_facecolor()) for r in _bar_rects(fig.axes[0])}
    assert to_hex(SUBSTRATE_COLOR) in colours
    assert to_hex(PRODUCT_COLOR) in colours


def test_panel_titles_name_the_strands():
    """The three panels are titled Combined / Forward / Reverse."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_bihistograms(result, sample=0)
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ['Combined', 'Forward', 'Reverse']


def test_leftmost_panel_labels_are_ca_state():
    """Row labels on the first panel are the CA-state motifs; others are blank."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_bihistograms(result, sample=0)
    left_labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert left_labels == list(FLANK16_LABELS_CA)
    # The other panels share the rows, so they carry no y labels.
    other_labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert set(other_labels) <= {''}


def test_rightmost_panel_has_ta_state_labels():
    """The rightmost panel carries the TA-state motif labels on its right side."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_bihistograms(result, sample=0)
    right_ax = fig.axes[2]
    texts = {t.get_text() for t in right_ax.texts}
    # Every TA-state motif label appears as free text on the rightmost panel.
    assert set(FLANK16_LABELS_TA) <= texts
    # The CA-state (substrate) labels are not repeated there.
    assert not (set(FLANK16_LABELS_CA) & texts)


def test_significant_motifs_are_marked():
    """A strongly divergent substrate/product pair gets a significance mark."""
    # Substrate concentrated in channel 0, product in channel 15, both well above
    # the min-sites gate, so the two flank distributions are maximally different.
    sub_fwd = np.zeros((16, 1))
    prod_fwd = np.zeros((16, 1))
    sub_fwd[0, 0] = 50.0
    prod_fwd[15, 0] = 50.0
    result = FlankSpectraResult(
        sub_fwd=sub_fwd,
        sub_rev=np.zeros((16, 1)),
        prod_fwd=prod_fwd,
        prod_rev=np.zeros((16, 1)),
        sample_names=['s'],
        n_skipped_flank=dict.fromkeys(
            ('sub_fwd', 'sub_rev', 'prod_fwd', 'prod_rev'), 0
        ),
    )
    fig = plot_flank_bihistograms(result, sample=0)
    # The forward panel (index 1) should carry at least one red '*' marker.
    fwd_ax = fig.axes[1]
    stars = [
        t
        for t in fwd_ax.texts
        if t.get_text() == '*' and to_hex(t.get_color()) == to_hex(SIG_COLOR)
    ]
    assert stars, 'expected a significance marker on a divergent channel'


def test_no_marks_when_below_min_sites():
    """Tiny counts never trigger a significance mark."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_bihistograms(result, sample=0)
    stars = [t for ax in fig.axes for t in ax.texts if t.get_text() == '*']
    assert stars == []


def test_bare_has_no_suptitle():
    """bare=True omits the caption/suptitle for embedding."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_bihistograms(result, sample=0, bare=True)
    assert fig._suptitle is None


def test_sample_out_of_range():
    """An out-of-range sample index raises IndexError."""
    result = make_result(['GCAT', 'GTAT'])
    with pytest.raises(IndexError):
        plot_flank_bihistograms(result, sample=9)


def test_negative_index_wraps():
    """A negative sample index selects from the end without error."""
    result = make_result(['GCAT', 'GTAT'])
    fig = plot_flank_bihistograms(result, sample=-1)
    assert len(fig.axes) == 3


def test_pooled_returns_three_panels():
    """The pooled overview figure also has three panels."""
    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    fig = plot_flank_bihistograms_pooled(result)
    assert len(fig.axes) == 3


def test_combined_only_single_panel():
    """Passing strands=('combined',) draws a single Combined panel."""
    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    fig = plot_flank_bihistograms_pooled(result, strands=('combined',))
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == 'Combined'
    # The one panel still carries both CA (left) and TA (right) labels.
    left = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert left == list(FLANK16_LABELS_CA)
    right = {t.get_text() for t in fig.axes[0].texts}
    assert set(FLANK16_LABELS_TA) <= right


def test_saves_to_file(tmp_path):
    """An outfile path writes a figure to disk."""
    result = make_result(['GCAT', 'GTAT'])
    out = tmp_path / 'flank.png'
    plot_flank_bihistograms(result, sample=0, outfile=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_percentage_normalises_each_state():
    """percentage=True rescales each state's bars to sum to 100% within a panel."""
    # A one-sample result with several populated substrate and product motifs.
    sub_fwd = np.zeros((16, 1))
    prod_fwd = np.zeros((16, 1))
    sub_fwd[[0, 5, 10], 0] = [10.0, 20.0, 30.0]
    prod_fwd[[1, 5], 0] = [4.0, 6.0]
    result = FlankSpectraResult(
        sub_fwd=sub_fwd,
        sub_rev=np.zeros((16, 1)),
        prod_fwd=prod_fwd,
        prod_rev=np.zeros((16, 1)),
        sample_names=['s'],
        n_skipped_flank=dict.fromkeys(
            ('sub_fwd', 'sub_rev', 'prod_fwd', 'prod_rev'), 0
        ),
    )
    fig = plot_flank_bihistograms(result, sample=0, percentage=True)
    ax = fig.axes[0]
    rects = _bar_rects(ax)
    substrate = sum(-r.get_width() for r in rects if r.get_width() < 0)
    product = sum(r.get_width() for r in rects if r.get_width() > 0)
    assert substrate == pytest.approx(100.0)
    assert product == pytest.approx(100.0)
    assert ax.get_xlabel() == '% of state'


def test_percentage_via_derip_method(mintest_path):
    """DeRIP.plot_flank_spectra(percentage=True) forwards the proportion option."""
    from derip2.derip import DeRIP

    d = DeRIP(mintest_path)
    d.calculate_rip()
    fig = d.plot_flank_spectra(percentage=True)
    assert fig.axes[0].get_xlabel() == '% of state'


# ---------------------------------------------------------------------------
# Conversion heatmap
# ---------------------------------------------------------------------------


def _compute_w(seqs, flank_length):
    """Compute flank spectra at a given flank width for a hand-built alignment."""
    align = MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=f'seq{i}') for i, s in enumerate(seqs)]
    )
    cls = classify_alignment(align, progress=False)
    return compute_flank_spectra(
        cls,
        sample_names=[f'seq{i}' for i in range(len(seqs))],
        flank_length=flank_length,
    )


def test_conversion_heatmap_1bp_has_image_and_colorbar():
    """The 1 bp conversion heatmap draws a 4x4 image plus a colorbar axes."""
    from derip2.plotting.flank_spectra import plot_flank_conversion_heatmap

    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    fig = plot_flank_conversion_heatmap(result, sample=None)
    # One image (the heatmap) on the main axes; a second axes for the colorbar.
    assert len(fig.axes) == 2
    main = fig.axes[0]
    assert len(main.images) == 1
    assert main.images[0].get_array().shape == (4, 4)


def test_conversion_heatmap_2bp_is_16x16():
    """A 2 bp flank produces a 16x16 conversion heatmap."""
    from derip2.plotting.flank_spectra import plot_flank_conversion_heatmap

    result = _compute_w(['ACCAGT', 'ACCAGT'], flank_length=2)
    fig = plot_flank_conversion_heatmap(result, sample=None)
    assert fig.axes[0].images[0].get_array().shape == (16, 16)


def test_bihistogram_rejects_wide_flank():
    """The 16-row bihistogram refuses a >1 bp flank result with a clear error."""
    result = _compute_w(['ACCAGT', 'ACCAGT'], flank_length=2)
    with pytest.raises(ValueError, match='1 bp flank'):
        plot_flank_bihistograms_pooled(result)
    with pytest.raises(ValueError, match='1 bp flank'):
        plot_flank_bihistograms(result, sample=0)


def test_heatmap_flank_sort_modes():
    """Proximal (default) orders the 5' axis by the base nearest the centre.

    For a 2 bp flank the upstream row labels are written outermost-base-first;
    'proximal' re-sorts them so the base nearest the centre is the primary key
    (reverse of the label), while 'alphabetical' keeps the plain label order.
    """
    from derip2.plotting.flank_spectra import plot_flank_conversion_heatmap

    result = _compute_w(['ACCAGT', 'ACCAGT'], flank_length=2)

    fig_alpha = plot_flank_conversion_heatmap(
        result, sample=None, flank_sort='alphabetical'
    )
    rows_alpha = [t.get_text() for t in fig_alpha.axes[0].get_yticklabels()]
    # Plain alphabetical of the 2 bp motif strings.
    assert rows_alpha == sorted(rows_alpha)

    fig_prox = plot_flank_conversion_heatmap(result, sample=None, flank_sort='proximal')
    rows_prox = [t.get_text() for t in fig_prox.axes[0].get_yticklabels()]
    # Sorted by the nearest-centre (last) base first, then the next base out.
    assert rows_prox == sorted(rows_prox, key=lambda s: s[::-1])
    # The nearest-5' base is now the primary grouping: first four rows all end 'A'.
    assert [r[-1] for r in rows_prox[:4]] == ['A', 'A', 'A', 'A']

    with pytest.raises(ValueError, match='flank_sort'):
        plot_flank_conversion_heatmap(result, sample=None, flank_sort='nonsense')


def test_heatmap_1bp_sort_modes_identical():
    """For a 1 bp flank the two sort modes give the same row order (A,C,G,T)."""
    from derip2.plotting.flank_spectra import plot_flank_conversion_heatmap

    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    prox = plot_flank_conversion_heatmap(result, sample=None, flank_sort='proximal')
    alpha = plot_flank_conversion_heatmap(
        result, sample=None, flank_sort='alphabetical'
    )
    rows_prox = [t.get_text() for t in prox.axes[0].get_yticklabels()]
    rows_alpha = [t.get_text() for t in alpha.axes[0].get_yticklabels()]
    assert rows_prox == rows_alpha == ['A', 'C', 'G', 'T']
