"""
Tests for the strand-bias figure.

These assert on artist geometry rather than comparing rendered images, which
would be brittle across matplotlib and font versions. The properties checked are
the ones the figure's meaning depends on: forward columns draw above the axis,
reverse columns below, the product segment touches the baseline, and masking
noise shrinks a bar without rescaling it.
"""

import logging

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from Bio.Align import MultipleSeqAlignment  # noqa: E402
from Bio.Seq import Seq  # noqa: E402
from Bio.SeqRecord import SeqRecord  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from derip2.aln_ops import classify_alignment  # noqa: E402
from derip2.plotting.logo import column_information  # noqa: E402
from derip2.plotting.strandbias import (  # noqa: E402
    BASE_COLORS,
    INK_MUTED,
    NOISE_ALPHA,
    ROLE_COLORS,
    _bar_path,
    _nice_ticks,
    _strand_direction,
    plot_strand_bias,
)

logging.disable(logging.CRITICAL)


@pytest.fixture(autouse=True)
def close_figures():
    """Never leak figures between tests."""
    yield
    plt.close('all')


def make_alignment(seqs):
    """Build a MultipleSeqAlignment from a list of sequence strings."""
    return MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=f'seq{i}') for i, s in enumerate(seqs)]
    )


# A forward RIP column at 0 (C/T followed by A) and a reverse RIP column at 3
# (G/A preceded by T), kept far enough apart not to interact.
FWD_ONLY = ['CACC', 'TACC', 'TACC']
REV_ONLY = ['CCTG', 'CCTA', 'CCTA']
MIXED = ['CACTG', 'TACTG', 'TACTA', 'CACTA']


def bar_patches(ax):
    """Stacked-bar segments only. Logo/consensus glyphs are drawn at zorder 3."""
    return [p for p in ax.patches if isinstance(p, PathPatch) and p.get_zorder() == 2]


def glyph_patches(ax):
    """Sequence-logo or consensus letter glyphs, and the gap dashes beside them."""
    return [p for p in ax.patches if isinstance(p, PathPatch) and p.get_zorder() == 3]


def region_bands(ax):
    """Column washes: the mutated-column highlight and the tie hatch."""
    return [p for p in ax.patches if isinstance(p, Rectangle)]


def patch_extent(patch):
    """(ymin, ymax) of a patch in data coordinates."""
    verts = patch.get_path().vertices
    return verts[:, 1].min(), verts[:, 1].max()


def patch_center_x(patch):
    """Bar centre, robust to the rounded corners on the outermost segment."""
    xs = patch.get_path().vertices[:, 0]
    return round(float((xs.min() + xs.max()) / 2.0), 3)


def total_height(ax):
    """Summed absolute height of every segment."""
    return sum(abs(hi - lo) for lo, hi in (patch_extent(p) for p in bar_patches(ax)))


# --------------------------------------------------------------------------
# Strand geometry
# --------------------------------------------------------------------------
def test_forward_columns_draw_above_the_axis():
    """A forward RIP column produces only non-negative bar geometry."""
    cls = classify_alignment(make_alignment(FWD_ONLY), progress=False)
    fig = plot_strand_bias(cls)
    ax = fig.axes[0]

    patches = bar_patches(ax)
    assert patches, 'expected at least one bar'
    for p in patches:
        lo, _hi = patch_extent(p)
        assert lo >= -1e-9, 'forward bar dipped below the axis'


def test_reverse_columns_draw_below_the_axis():
    """A reverse RIP column produces only non-positive bar geometry."""
    cls = classify_alignment(make_alignment(REV_ONLY), progress=False)
    fig = plot_strand_bias(cls)
    ax = fig.axes[0]

    patches = bar_patches(ax)
    assert patches, 'expected at least one bar'
    for p in patches:
        _lo, hi = patch_extent(p)
        assert hi <= 1e-9, 'reverse bar rose above the axis'


def test_mixed_alignment_draws_on_both_arms():
    """An alignment RIP'd on both strands puts bars above and below."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    fig = plot_strand_bias(cls)
    ax = fig.axes[0]

    extents = [patch_extent(p) for p in bar_patches(ax)]
    assert any(hi > 1e-9 for _lo, hi in extents)
    assert any(lo < -1e-9 for lo, _hi in extents)


def test_product_segment_touches_the_baseline():
    """The RIP product is drawn against zero so products share a baseline."""
    cls = classify_alignment(make_alignment(FWD_ONLY), progress=False)
    fig = plot_strand_bias(cls, xaxis='none')
    ax = fig.axes[0]

    product_color = matplotlib.colors.to_hex(BASE_COLORS['T'])
    products = [
        p
        for p in bar_patches(ax)
        if matplotlib.colors.to_hex(p.get_facecolor()) == product_color
    ]
    assert products
    for p in products:
        lo, _hi = patch_extent(p)
        assert lo == pytest.approx(0.0, abs=1e-9)


def test_gutter_offsets_bars_away_from_the_lettering():
    """With an x-axis sequence, bars start at the gutter edge, not at zero."""
    cls = classify_alignment(make_alignment(FWD_ONLY), progress=False)
    fig = plot_strand_bias(cls, xaxis='logo')
    ax = fig.axes[0]

    lows = [patch_extent(p)[0] for p in bar_patches(ax)]
    assert min(lows) > 0.05, 'bars should clear the lettering band'


# --------------------------------------------------------------------------
# Scaling and stack composition
# --------------------------------------------------------------------------
def test_full_bar_sums_to_one_under_column_scaling():
    """With every base stacked, a column's segments fill exactly its own depth."""
    cls = classify_alignment(make_alignment(FWD_ONLY), progress=False)
    fig = plot_strand_bias(cls, scale='column', stack='all', columns='all')
    ax = fig.axes[0]

    # Group segment heights by bar centre.
    by_column = {}
    for p in bar_patches(ax):
        x = patch_center_x(p)
        lo, hi = patch_extent(p)
        by_column.setdefault(x, 0.0)
        by_column[x] += abs(hi - lo)

    for x, height in by_column.items():
        assert height == pytest.approx(1.0, abs=1e-6), f'column at {x} did not fill'


def test_dropping_noise_shrinks_bars_without_rescaling():
    """Omitting noise bases removes height; the remainder keeps its scale."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)

    ax_signal = plot_strand_bias(cls, stack='signal', columns='all').axes[0]
    ax_full = plot_strand_bias(cls, stack='all', columns='all').axes[0]

    signal, full = total_height(ax_signal), total_height(ax_full)
    assert signal < full, 'omitting noise must remove height'

    # The retained product/substrate segments keep exactly the heights they had.
    def signal_heights(ax):
        noise = matplotlib.colors.to_hex(ROLE_COLORS['other'])
        return sorted(
            round(abs(hi - lo), 6)
            for lo, hi in (
                patch_extent(p)
                for p in bar_patches(ax)
                if matplotlib.colors.to_hex(p.get_facecolor()) != noise
            )
        )

    assert signal_heights(ax_signal) == signal_heights(ax_full)


def test_product_stack_draws_the_product_alone():
    """stack='product' leaves one segment per bar, in the product colour."""
    cls = classify_alignment(make_alignment(FWD_ONLY), progress=False)
    ax = plot_strand_bias(cls, stack='product', columns='all').axes[0]

    patches = bar_patches(ax)
    assert len(patches) == 1, 'only the one forward column carries a product'
    assert matplotlib.colors.to_hex(patches[0].get_facecolor()) == (
        matplotlib.colors.to_hex(BASE_COLORS['T'])
    )


def test_noise_segment_is_translucent_under_full_stack():
    """Noise recedes within a bar whose product and substrate stay opaque."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    ax = plot_strand_bias(cls, stack='all', columns='all', emphasis=False).axes[0]

    noise = matplotlib.colors.to_hex(ROLE_COLORS['other'])
    by_role = {}
    for p in bar_patches(ax):
        is_noise = matplotlib.colors.to_hex(p.get_facecolor()) == noise
        by_role.setdefault(is_noise, set()).add(p.get_alpha())

    assert by_role[False] == {1.0}
    assert by_role[True] == {NOISE_ALPHA}


def test_alignment_scale_shortens_gappy_columns():
    """Under 'alignment' scaling a mostly-gap column draws a short bar."""
    # Column 0 is C/T in only two of six rows.
    align = make_alignment(['CACC', 'TACC', '-ACC', '-ACC', '-ACC', '-ACC'])
    cls = classify_alignment(align, progress=False)

    window = (0, 1)
    ax_col = plot_strand_bias(
        cls, scale='column', stack='all', column_range=window
    ).axes[0]
    ax_aln = plot_strand_bias(
        cls, scale='alignment', stack='all', column_range=window
    ).axes[0]

    assert total_height(ax_aln) < total_height(ax_col)
    # Two of six sequences carry a base, so the bar reaches 1/3 of full height.
    assert total_height(ax_aln) == pytest.approx(2.0 / 6.0, abs=1e-6)


def test_counts_scale_uses_raw_sequence_counts():
    """'counts' leaves the y-axis in sequences, not proportions."""
    cls = classify_alignment(make_alignment(FWD_ONLY), progress=False)
    fig = plot_strand_bias(cls, scale='counts', stack='all', columns='all')
    ax = fig.axes[0]
    assert ax.get_ylabel() == 'Sequences'
    assert total_height(ax) == pytest.approx(3.0 * 4, abs=1e-6)  # 3 seqs x 4 cols


# --------------------------------------------------------------------------
# Column selection
# --------------------------------------------------------------------------
def test_column_range_restricts_the_plot():
    """column_range windows the alignment without changing classification."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    full = plot_strand_bias(cls, columns='all').axes[0]
    windowed = plot_strand_bias(cls, columns='all', column_range=(0, 2)).axes[0]
    assert len(bar_patches(windowed)) < len(bar_patches(full))


def test_long_alignments_are_drawn_in_full_by_default():
    """No column cap: the figure widens instead of refusing to draw."""
    cls = classify_alignment(make_alignment(['CA' * 40, 'TA' * 40]), progress=False)
    fig = plot_strand_bias(cls)
    assert fig.get_size_inches()[0] > 6.0


def test_explicit_max_columns_still_refuses():
    """A caller who sets a cap gets it enforced, with an explanation."""
    cls = classify_alignment(make_alignment(['CA' * 40, 'TA' * 40]), progress=False)
    with pytest.raises(ValueError, match='max_columns'):
        plot_strand_bias(cls, max_columns=5)


@pytest.mark.parametrize('columns', ['rip', 'substrate', 'all'])
def test_every_column_with_a_base_is_drawn(columns):
    """`columns` no longer selects bars: every column carries one."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    ax = plot_strand_bias(cls, columns=columns, stack='all').axes[0]
    drawn = {patch_center_x(p) for p in bar_patches(ax)}
    assert drawn == {float(x) for x in np.where(cls.base_count > 0)[0]}


def test_tied_column_is_hatched_not_dropped():
    """A column with no strand majority is hatched, never ticked on the axis."""
    # Column 0 has two C and two G: neither strand holds a strict majority.
    align = make_alignment(['CA', 'CA', 'GT', 'GT'])
    cls = classify_alignment(align, progress=False)
    assert _strand_direction(cls)[0] == 0

    ax = plot_strand_bias(cls, columns='all').axes[0]

    markers = [ln for ln in ax.lines if ln.get_marker() not in (None, 'None', '')]
    assert not markers, 'nothing may be marked on the central axis'

    # Both columns are tied here: C/C/G/G at 0, and A/A/T/T at 1.
    hatched = sorted(p.get_x() for p in region_bands(ax) if p.get_hatch())
    assert hatched == pytest.approx([-0.5, 0.5])
    assert not bar_patches(ax), 'a tied column carries no bar'


# --------------------------------------------------------------------------
# Lettering
# --------------------------------------------------------------------------
def glyph_center_x(patch):
    """Alignment column a letter or dash is centred on."""
    # A glyph path carries Bezier control points outside its drawn extent, so
    # its centre must come from the true extents rather than the raw vertices.
    box = patch.get_path().get_extents()
    return round(box.x0 + box.width / 2, 3)


def glyph_columns(ax):
    """Alignment column of every letter drawn, with its opacity."""
    return {glyph_center_x(p): p.get_alpha() for p in glyph_patches(ax)}


def test_all_columns_letters_every_position():
    """columns='all' letters every position holding a base."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    ax = plot_strand_bias(
        cls, columns='all', xaxis='derip', consensus_seq='CACTG'
    ).axes[0]
    assert set(glyph_columns(ax)) == {0.0, 1.0, 2.0, 3.0, 4.0}


def test_rip_letters_the_site_and_its_dinucleotide_partner():
    """A RIP site is lettered together with the partner base of its motif."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    ax = plot_strand_bias(
        cls, columns='rip', xaxis='derip', consensus_seq='CACTG'
    ).axes[0]

    # Forward site at column 0 (C/T before an A) pairs with the A at column 1.
    # Reverse site at column 4 (G/A after a T) pairs with the T at column 3.
    assert set(glyph_columns(ax)) == {0.0, 1.0, 3.0, 4.0}


def test_partner_letters_recede_behind_their_site():
    """The context half of a motif is drawn fainter than the deaminated site."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    ax = plot_strand_bias(
        cls, columns='rip', xaxis='derip', consensus_seq='CACTG'
    ).axes[0]

    alphas = glyph_columns(ax)
    assert alphas[1.0] < alphas[0.0]  # forward partner < forward site
    assert alphas[3.0] < alphas[4.0]  # reverse partner < reverse site


def test_substrate_letters_only_untouched_substrate_columns():
    """columns='substrate' skips RIP-like columns and their products."""
    # Column 0 is a RIP-like forward column (C and T, both before an A).
    # Column 2 is 100% CpA substrate: no product anywhere, so RIP never fired.
    cls = classify_alignment(make_alignment(['CACA', 'TACA']), progress=False)
    assert cls.fwd_col[0] and not cls.fwd_col[2]

    ax = plot_strand_bias(
        cls, columns='substrate', xaxis='derip', consensus_seq='CACA'
    ).axes[0]
    # The untouched substrate at column 2 and its partner A at column 3.
    assert set(glyph_columns(ax)) == {2.0, 3.0}


def test_gapped_consensus_renders_a_dash():
    """A gap in the consensus is drawn as a dash, never as a borrowed base."""
    # Column 0 holds a lone C among gaps, so the deRIP'd consensus gaps it out.
    cls = classify_alignment(
        make_alignment(['CACC', '-ACC', '-ACC', '-ACC']), progress=False
    )
    ax = plot_strand_bias(cls, columns='all', xaxis='derip', consensus_seq='-ACC').axes[
        0
    ]

    at_gap = [p for p in glyph_patches(ax) if glyph_center_x(p) == 0.0]
    assert len(at_gap) == 1, 'gapped consensus column lost its mark'

    dash = at_gap[0]
    assert matplotlib.colors.to_hex(dash.get_facecolor()) == (
        matplotlib.colors.to_hex(INK_MUTED)
    )
    # A dash is a thin rule, not a letter filling the gutter band.
    lo, hi = patch_extent(dash)
    letters = [p for p in glyph_patches(ax) if glyph_center_x(p) == 1.0]
    letter_lo, letter_hi = patch_extent(letters[0])
    assert (hi - lo) < 0.25 * (letter_hi - letter_lo)


def test_logo_axis_stacks_a_glyph_per_observed_base():
    """A logo column carries one glyph for each base present in that column."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    logo = plot_strand_bias(cls, columns='rip', xaxis='logo').axes[0]

    lettered = np.array([0, 1, 3, 4])
    expected = sum(int((cls.base_counts[c, :4] > 0).sum()) for c in lettered)
    assert len(glyph_patches(logo)) == expected


# --------------------------------------------------------------------------
# Axis decoration
# --------------------------------------------------------------------------
def test_no_glyphs_without_an_xaxis_sequence():
    """xaxis='none' draws bars only, and lettering never disturbs the bars."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    plain = plot_strand_bias(cls, xaxis='none').axes[0]
    lettered = plot_strand_bias(cls, xaxis='derip', consensus_seq='CACTG').axes[0]

    assert glyph_patches(plain) == []
    assert len(bar_patches(lettered)) == len(bar_patches(plain))


def test_emphasis_washes_only_mutated_columns():
    """The highlight band marks the columns the mode saw a transition in."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)

    def washes(ax):
        # The tie hatch is a band too; the highlight is the one without a hatch.
        return [p for p in region_bands(ax) if not p.get_hatch()]

    mutated = int(((cls.prod_fwd.sum(axis=0) + cls.prod_rev.sum(axis=0)) > 0).sum())
    assert len(washes(plot_strand_bias(cls).axes[0])) == mutated
    assert washes(plot_strand_bias(cls, emphasis=False).axes[0]) == []


def test_chrome_is_publication_styled():
    """Centred heading, no gridlines, left and bottom spines only."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    ax = plot_strand_bias(cls).axes[0]

    assert ax.title.get_ha() == 'center'
    assert not any(line.get_visible() for line in ax.get_ygridlines())
    assert ax.spines['left'].get_visible() and ax.spines['bottom'].get_visible()
    assert not ax.spines['top'].get_visible()
    assert not ax.spines['right'].get_visible()


def test_derip_axis_requires_consensus():
    """Asking for the deRIP'd sequence without supplying it is an error."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    with pytest.raises(ValueError, match='requires consensus_seq'):
        plot_strand_bias(cls, xaxis='derip')


def test_legend_order_is_fixed():
    """Legend entries are ordered forward pair, reverse pair, then noise."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    ax = plot_strand_bias(cls, stack='all', columns='all').axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [
        'T (product)',
        'C (substrate)',
        'A (product)',
        'G (substrate)',
        'Other bases',
        'Mutated column',
    ]


def test_noise_legend_entry_only_under_the_full_stack():
    """'Other bases' is named only when the noise segment is actually drawn."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)

    def labels(**kwargs):
        ax = plot_strand_bias(cls, columns='all', **kwargs).axes[0]
        return [t.get_text() for t in ax.get_legend().get_texts()]

    assert 'Other bases' not in labels(stack='signal')
    assert 'Other bases' not in labels(stack='product')
    assert 'Other bases' in labels(stack='all')


def test_tied_column_is_named_in_the_legend():
    """A hatched band is meaningless without a key."""
    cls = classify_alignment(make_alignment(['CA', 'CA', 'GT', 'GT']), progress=False)
    ax = plot_strand_bias(cls, columns='all').axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert 'No strand majority' in labels


def test_role_coloring_uses_semantic_labels():
    """color_by='role' names the role rather than the base."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    ax = plot_strand_bias(cls, color_by='role', emphasis=False).axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ['RIP product', 'RIP substrate']


@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({'mode': 'bogus'}, 'mode must be one of'),
        ({'scale': 'bogus'}, 'scale must be one of'),
        ({'xaxis': 'bogus'}, 'xaxis must be one of'),
        ({'color_by': 'bogus'}, 'color_by must be one of'),
        ({'columns': 'bogus'}, 'columns must be one of'),
        ({'stack': 'bogus'}, 'stack must be one of'),
    ],
)
def test_invalid_options_raise(kwargs, match):
    """Unknown options are rejected rather than silently defaulting."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    with pytest.raises(ValueError, match=match):
        plot_strand_bias(cls, **kwargs)


def test_writes_svg(tmp_path):
    """A vector file is produced for publication use."""
    cls = classify_alignment(make_alignment(MIXED), progress=False)
    out = tmp_path / 'bias.svg'
    plot_strand_bias(cls, outfile=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert out.read_text().lstrip().startswith(('<?xml', '<svg'))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def test_bar_path_rounding_is_clamped_to_the_bar():
    """A radius larger than the bar cannot invert or overshoot its geometry."""
    path = _bar_path(0.0, 0.0, width=0.5, height=0.01, direction=1, radius=10.0)
    ys = path.vertices[:, 1]
    xs = path.vertices[:, 0]
    assert ys.min() == pytest.approx(0.0)
    assert ys.max() == pytest.approx(0.01)
    assert xs.min() == pytest.approx(-0.25)
    assert xs.max() == pytest.approx(0.25)


def test_bar_path_grows_downward_for_reverse_strand():
    """direction=-1 places the data end below the baseline."""
    path = _bar_path(0.0, 0.0, width=0.5, height=0.4, direction=-1, radius=0.0)
    assert path.vertices[:, 1].min() == pytest.approx(-0.4)
    assert path.vertices[:, 1].max() == pytest.approx(0.0)


@pytest.mark.parametrize('extent', [0.05, 0.33, 1.0, 2.5])
def test_nice_ticks_span_the_data(extent):
    """Ticks start at zero and reach at least the largest bar."""
    ticks = _nice_ticks(extent, 'column')
    assert ticks[0] == 0.0
    assert ticks[-1] >= extent - 1e-9
    assert np.all(np.diff(ticks) > 0)


def test_nice_ticks_are_integers_for_counts():
    """Counting sequences never yields a fractional tick."""
    ticks = _nice_ticks(7, 'counts')
    assert np.all(ticks == ticks.astype(int))


def test_column_information_is_two_bits_for_an_invariant_column():
    """A column with a single base carries the maximum 2 bits."""
    info, freqs = column_information([0, 4, 0, 0])  # all C
    assert info == pytest.approx(2.0)
    assert freqs[1] == pytest.approx(1.0)


def test_column_information_is_zero_for_a_uniform_column():
    """Equal base frequencies carry no information."""
    info, _ = column_information([2, 2, 2, 2])
    assert info == pytest.approx(0.0)


def test_column_information_handles_empty_columns():
    """An all-gap column has no information and no frequencies."""
    info, freqs = column_information([0, 0, 0, 0])
    assert info == 0.0
    assert np.all(freqs == 0)
