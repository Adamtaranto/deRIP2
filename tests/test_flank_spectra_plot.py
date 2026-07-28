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


def test_conversion_heatmap_uses_the_shared_conversion_ramp():
    """The heatmap draws with CONVERSION_CMAP, not some other colormap."""
    from derip2.plotting.flank_spectra import plot_flank_conversion_heatmap
    from derip2.plotting.persequence import CONVERSION_CMAP

    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    fig = plot_flank_conversion_heatmap(result, sample=None)
    image = fig.axes[0].images[0]
    assert image.cmap is CONVERSION_CMAP
    # The scale is pinned to 0-100 % so colour is comparable between figures.
    assert (image.norm.vmin, image.norm.vmax) == (0, 100)


def _lstar(rgb):
    """CIE L* of one or many sRGB triples in [0, 1]."""
    import numpy as np

    rgb = np.asarray(rgb, dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(
        luminance > 0.008856, 116 * np.cbrt(luminance) - 16, 903.3 * luminance
    )


def test_conversion_ramp_is_viridis_dark_low_light_high():
    """The default ramp is viridis: dark purple low, bright yellow high.

    A reversed viridis would invert the figure's meaning while still passing
    every structural check, so pin the orientation, not just the name.

    Stated as lightness rather than hue, because that is the property doing the
    work and it survives a future swap to a differently-hued sequential map.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from derip2.plotting.persequence import CONVERSION_CMAP, CONVERSION_CMAP_NAME

    assert CONVERSION_CMAP_NAME == 'viridis'
    samples = np.linspace(0, 1, 64)
    assert np.allclose(CONVERSION_CMAP(samples), plt.get_cmap('viridis')(samples))

    # 0 % is dark, 100 % is light, by a wide margin.
    assert _lstar(CONVERSION_CMAP(0.0)[:3]) < 30
    assert _lstar(CONVERSION_CMAP(1.0)[:3]) > 80


def test_conversion_ramp_lightness_is_monotonic():
    """CIE L* changes monotonically across the default ramp.

    A sequential map keeps the cells ordered in a greyscale reproduction, which
    the diverging defaults this replaced could not do (their lightness peaked at
    the midpoint). Guards against a swap back to a map without the property.

    Direction-agnostic on purpose: whether the ramp runs light-to-dark or
    dark-to-light is the orientation test's business, and asserting it here too
    would mean editing two tests every time the default is flipped.
    """
    import numpy as np

    from derip2.plotting.persequence import CONVERSION_CMAP

    lstar = _lstar(CONVERSION_CMAP(np.linspace(0, 1, 256))[:, :3])
    steps = np.diff(lstar)
    # Strictly monotonic one way or the other, with a small tolerance for
    # sampling wobble.
    assert np.all(steps < 1e-6) or np.all(steps > -1e-6)
    # And a wide span, so the two ends are clearly distinguishable.
    assert abs(lstar[-1] - lstar[0]) > 60


def test_no_data_colour_is_distinct_from_the_whole_ramp():
    """A motif seen zero times cannot be mistaken for a real conversion value.

    Not just the ends: the no-data colour has to sit off the ramp at *every*
    level. This is why it is a neutral grey rather than the figure surface —
    YlOrBr starts at a near-white #ffffe5, only ~0.09 from the surface, and on
    the 3 bp grid roughly half the cells are blank. A future default that itself
    passes through a light grey (coolwarm's midpoint, say) would trip this.
    """
    import numpy as np

    from derip2.plotting.persequence import (
        CONVERSION_CMAP,
        NO_DATA_COLOR,
        _hex_to_rgb,
    )

    blank = np.array(CONVERSION_CMAP(np.nan)[:3])
    assert np.allclose(blank, _hex_to_rgb(NO_DATA_COLOR), atol=0.01)

    ramp = CONVERSION_CMAP(np.linspace(0, 1, 256))[:, :3]
    closest = float(np.linalg.norm(ramp - blank, axis=1).min())
    assert closest > 0.20, (
        f'the no-data colour is only {closest:.3f} from the closest ramp colour'
    )


# Machado et al. 2009 severity-1.0 colour-vision-deficiency simulation matrices.
_CVD_MATRICES = {
    'deuteranopia': [
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881],
    ],
    'protanopia': [
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ],
    'tritanopia': [
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900],
    ],
}


def _worst_cvd_confusion(cmap, min_gap=0.20, n=21):
    """
    Closest simulated colour pair among values at least ``min_gap`` apart.

    Comparing only *adjacent* samples is the wrong measurement for a diverging
    map, and would pass a map that is genuinely unsafe: on a ramp that runs out
    to a neutral midpoint and back, each small step is distinct while values
    symmetrically either side of the midpoint collapse onto the same simulated
    colour. Scanning all well-separated pairs is what catches that.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The ramp to probe.
    min_gap : float, optional
        Only compare values at least this far apart on the 0-1 scale, so a
        genuinely fine distinction is not counted as a failure (default 0.20).
    n : int, optional
        Number of samples across the ramp (default 21).

    Returns
    -------
    float
        The smallest simulated sRGB distance found, over every simulated form of
        colour vision deficiency.
    """
    import numpy as np

    levels = np.linspace(0, 1, n)
    worst = np.inf
    for matrix in _CVD_MATRICES.values():
        simulated = np.array(
            [np.clip(np.asarray(matrix) @ np.array(cmap(v)[:3]), 0, 1) for v in levels]
        )
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                if levels[j] - levels[i] < min_gap:
                    continue
                worst = min(worst, float(np.linalg.norm(simulated[j] - simulated[i])))
    return worst


def test_default_ramp_is_colourblind_safe():
    """Well-separated values stay well-separated under simulated CVD.

    Blue-to-red through a neutral is the colourblind-safe diverging axis, and
    this is the assertion that makes that claim mean something: no two values
    20 percentage points or more apart may simulate to near-identical colours.

    Measured for reference: YlOrBr (the default) 0.17, coolwarm 0.24, cividis
    0.22, magma_r 0.21, viridis 0.18. An earlier Spectral_r default scored 0.06
    — its green and orange arms collapse onto each other — which is exactly what
    ColorBrewer's "not colourblind safe" label on Spectral refers to, and the
    gap between that and everything else is what this threshold sits in.
    """
    from derip2.plotting.persequence import CONVERSION_CMAP

    assert _worst_cvd_confusion(CONVERSION_CMAP) > 0.15


def test_perceptually_uniform_alternatives_are_also_safe():
    """The maps the docs recommend for greyscale also survive the CVD check.

    coolwarm's lightness peaks at its midpoint, so greyscale cannot order the
    cells; the docs point at 'magma_r', 'cividis' and 'viridis' for that case.
    Those are only useful advice if they are themselves colourblind-safe.

    The bar is lower than the default's because these score slightly lower on
    this metric (measured: coolwarm 0.24, cividis 0.22, magma_r 0.21,
    viridis 0.18) — all comfortably distinguishable, none as separated as the
    default. For contrast, Spectral_r scores 0.06.
    """
    from derip2.plotting.persequence import resolve_cmap

    for name in ('magma_r', 'cividis', 'viridis'):
        assert _worst_cvd_confusion(resolve_cmap(name)) > 0.15, name


def test_resolve_cmap_accepts_the_documented_forms():
    """A name, a Colormap and a colour list all resolve to a colormap."""
    import matplotlib.pyplot as plt
    import numpy as np

    from derip2.plotting.persequence import CONVERSION_CMAP, resolve_cmap

    # None falls back to the package default.
    assert resolve_cmap() is CONVERSION_CMAP
    assert resolve_cmap(None) is CONVERSION_CMAP

    # A registered name, including the reversed form.
    by_name = resolve_cmap('viridis')
    samples = np.linspace(0, 1, 32)
    assert np.allclose(by_name(samples), plt.get_cmap('viridis')(samples))
    assert not np.allclose(
        resolve_cmap('viridis_r')(samples), plt.get_cmap('viridis')(samples)
    )

    # A Colormap instance is taken as given.
    assert np.allclose(
        resolve_cmap(plt.get_cmap('plasma'))(samples), plt.get_cmap('plasma')(samples)
    )

    # A colour list is interpolated low value first.
    custom = resolve_cmap(['#ffffff', '#000000'])
    assert np.allclose(custom(0.0)[:3], (1.0, 1.0, 1.0), atol=0.01)
    assert np.allclose(custom(1.0)[:3], (0.0, 0.0, 0.0), atol=0.01)


def test_resolve_cmap_sets_the_no_data_colour_on_every_form():
    """However the palette arrives, empty cells take the same no-data colour."""
    import matplotlib.pyplot as plt
    import numpy as np

    from derip2.plotting.persequence import NO_DATA_COLOR, _hex_to_rgb, resolve_cmap

    for spec in ('viridis', plt.get_cmap('plasma'), ['#ffffff', '#000000']):
        resolved = resolve_cmap(spec)
        assert np.allclose(resolved(np.nan)[:3], _hex_to_rgb(NO_DATA_COLOR), atol=0.01)


def test_resolve_cmap_rejects_bad_input():
    """Bad palettes fail with a message naming what was wrong."""
    from derip2.plotting.persequence import resolve_cmap

    with pytest.raises(ValueError, match='Unknown matplotlib colormap'):
        resolve_cmap('not-a-real-colormap')
    with pytest.raises(ValueError, match='at least two colours'):
        resolve_cmap(['#ffffff'])
    with pytest.raises(ValueError, match='Invalid colour at position 1'):
        resolve_cmap(['#ffffff', 'definitely-not-a-colour'])
    with pytest.raises(TypeError, match='must be a colormap name'):
        resolve_cmap(42)


def test_conversion_heatmap_honours_a_custom_cmap():
    """The cmap argument reaches the drawn image, overriding the default."""
    import matplotlib.pyplot as plt
    import numpy as np

    from derip2.plotting.flank_spectra import plot_flank_conversion_heatmap
    from derip2.plotting.persequence import CONVERSION_CMAP

    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])

    fig = plot_flank_conversion_heatmap(result, sample=None, cmap='magma_r')
    image = fig.axes[0].images[0]
    samples = np.linspace(0, 1, 32)
    assert np.allclose(image.cmap(samples), plt.get_cmap('magma_r')(samples))
    assert not np.allclose(image.cmap(samples), CONVERSION_CMAP(samples))

    # A colour list works the same way.
    fig = plot_flank_conversion_heatmap(
        result, sample=None, cmap=['#ffffff', '#2a78d6']
    )
    assert np.allclose(fig.axes[0].images[0].cmap(0.0)[:3], (1.0, 1.0, 1.0), atol=0.01)


def test_conversion_heatmap_rejects_a_bad_cmap_before_drawing():
    """An unusable palette raises rather than half-building a figure."""
    from derip2.plotting.flank_spectra import plot_flank_conversion_heatmap

    result = make_result(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    before = len(plt.get_fignums())
    with pytest.raises(ValueError, match='Unknown matplotlib colormap'):
        plot_flank_conversion_heatmap(result, sample=None, cmap='nope')
    assert len(plt.get_fignums()) == before, 'a figure was left open on failure'


def test_derip_method_forwards_cmap(mintest_path):
    """DeRIP.plot_flank_conversion_heatmap passes cmap through to the plot."""
    import matplotlib.pyplot as plt
    import numpy as np

    from derip2.derip import DeRIP

    derip = DeRIP(mintest_path)
    derip.calculate_rip()
    fig = derip.plot_flank_conversion_heatmap(cmap='cividis')
    samples = np.linspace(0, 1, 32)
    assert np.allclose(
        fig.axes[0].images[0].cmap(samples), plt.get_cmap('cividis')(samples)
    )


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
