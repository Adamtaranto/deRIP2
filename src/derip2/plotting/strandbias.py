"""
Diverging stacked-bar figures of per-column RIP strand bias.

Each alignment column that carries RIP signal becomes one bar. The bar is drawn
**above** the axis when the deamination is observed on the forward strand
(C→T in CpA context) and **below** when it is observed on the reverse strand
(G→A, seen on the forward strand as the loss of a TpG).

A bar therefore sits at the column where the deaminated base itself lies. A
forward event is scored at the C's column; the reverse event of the same duplex
is scored at the G's column, one position to the right. Bars for a single
physical TpA dinucleotide can consequently appear in adjacent columns on
opposite sides of the axis — this is the strand ambiguity, made visible.

Within a bar the RIP **product** segment is drawn against the zero line and the
unmutated **substrate** stacks outward. Every product segment therefore shares a
common baseline, so the extent of RIP can be compared across columns at a
glance.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Base identity palette, validated for colourblind separation against a light
# surface: worst all-pairs Machado-2009 dE is 13.3 (target >= 12), every slot
# clears 3:1 contrast and sits inside the OKLCH lightness band.
#
# C, T and A keep their conventional colours. G is violet rather than the usual
# yellow because no yellow step reaches 3:1 contrast on a light surface.
BASE_COLORS = {
    'A': '#008300',  # green
    'C': '#2a78d6',  # blue
    'G': '#4a3aa7',  # violet
    'T': '#e34948',  # red
}

# Semantic palette: encodes the role a base plays rather than its identity.
# The noise grey is darker than the column wash it is drawn over, so a
# translucent noise segment can never be mistaken for the wash behind it.
ROLE_COLORS = {
    'product': '#2a78d6',  # blue
    'substrate': '#eb6834',  # orange
    'other': '#a9a79d',  # muted grey - bases carrying no RIP signal
}

# Chart chrome. Figures render on their own light surface in both themes, as a
# printed figure does; the validated palette is only guaranteed against it.
SURFACE = '#fcfcfb'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'

# Wash drawn behind a column in which the current mode observed a transition.
# Hueless by design: any tint here would compete with the nucleotide palette.
HIGHLIGHT = '#d8d6cd'
HIGHLIGHT_ALPHA = 0.42

# Near-black used for the spines and ticks, as journal figures expect.
AXIS_INK = '#1a1a1a'

MODES = ('rip', 'non_rip', 'all_deamination')
SCALES = ('column', 'alignment', 'counts')
XAXIS_STYLES = ('none', 'logo', 'derip')
COLOR_BY = ('base', 'role')
COLUMN_SETS = ('rip', 'substrate', 'all')
STACK_SETS = ('signal', 'product', 'all')

MODE_TITLES = {
    'rip': 'RIP-like mutations',
    'non_rip': 'Non-RIP deamination',
    'all_deamination': 'All C/G deamination',
}

# Gutter half-height, in y-units, reserved around the zero line for lettering.
# The logo needs more room than a single consensus character because it stacks
# up to four glyphs.
GUTTER = {'none': 0.0, 'derip': 0.11, 'logo': 0.20}

# Opacity of the lettering roles. A partner letter is the second half of a
# dinucleotide (the A of CpA, the T of TpG); it names the context, not the site,
# so it recedes.
SITE_ALPHA = 1.0
CONTEXT_ALPHA = 0.72
PARTNER_ALPHA = 0.5
PARTNER_SCALE = 0.8

# Opacity of a bar in a column where the current mode saw no transition, and of
# the noise segment under ``stack='all'``.
CONTEXT_BAR_ALPHA = 0.35
NOISE_ALPHA = 0.5

# Typography, in points. Journal figures are set small and read at column width.
FONT_STACK = ['Arial', 'Helvetica', 'DejaVu Sans']
TITLE_SIZE = 9
AXIS_LABEL_SIZE = 8
TICK_LABEL_SIZE = 7
LEGEND_SIZE = 7
ANNOTATION_SIZE = 6.5

# Headroom above the tallest bar / top tick on each arm, as a multiplier of the
# extent. The arm labels ("forward strand ...", "reverse strand ...") are pinned
# to the axes top/bottom edges, so the bars need clearance beneath them or the
# labels sit on top of the tallest bars.
ARM_LABEL_HEADROOM = 1.22

# Hatch marking a column in which neither strand holds a majority.
TIE_HATCH = '///'

# Beyond this the matplotlib canvas exceeds its pixel ceiling.
MAX_FIG_INCHES = 200.0
RASTER_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def _column_segments(cls, mode):
    """
    Per-column product / substrate counts for each strand under a display mode.

    Parameters
    ----------
    cls : derip2.aln_ops.ColumnClassification
        Classification of the alignment.
    mode : {'rip', 'non_rip', 'all_deamination'}
        Which deamination events to display.

        - ``'rip'``: products are T in TpA context inside a RIP column;
          substrates are C in CpA context.
        - ``'non_rip'``: products are T outside TpA context; substrates are C
          outside CpA context.
        - ``'all_deamination'``: every T and every C, regardless of context.

    Returns
    -------
    dict
        ``fwd_product``, ``fwd_substrate``, ``rev_product``, ``rev_substrate``,
        each an ``(n_cols,)`` int array.

    Raises
    ------
    ValueError
        If ``mode`` is not recognised.
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, got {mode!r}')

    is_C = cls.arr == b'C'
    is_T = cls.arr == b'T'
    is_G = cls.arr == b'G'
    is_A = cls.arr == b'A'

    if mode == 'rip':
        fwd_p, fwd_s = cls.prod_fwd, cls.sub_fwd
        rev_p, rev_s = cls.prod_rev, cls.sub_rev
    elif mode == 'non_rip':
        # Deamination outside RIP dinucleotide context: the T is not in TpA,
        # and the surviving C is not in CpA.
        fwd_p, fwd_s = cls.nonrip_fwd, is_C & ~cls.ca
        rev_p, rev_s = cls.nonrip_rev, is_G & ~cls.tg
    else:  # 'all_deamination'
        fwd_p, fwd_s = is_T, is_C
        rev_p, rev_s = is_A, is_G

    return {
        'fwd_product': fwd_p.sum(axis=0),
        'fwd_substrate': fwd_s.sum(axis=0),
        'rev_product': rev_p.sum(axis=0),
        'rev_substrate': rev_s.sum(axis=0),
    }


def _partner_columns(cls, site_mask, forward):
    """
    Locate the column holding the second base of each site column's dinucleotide.

    A RIP motif spans two alignment columns: the deaminated site and the base
    that gives it its context — the ``A`` downstream of a ``CpA``, the ``T``
    upstream of a ``TpG``. Because dinucleotides are defined per row over the
    nearest *non-gap* neighbour, rows can disagree on which column holds the
    partner. The plurality choice is taken.

    Parameters
    ----------
    cls : derip2.aln_ops.ColumnClassification
        Classification of the alignment.
    site_mask : numpy.ndarray
        ``(n_cols,)`` boolean mask of the site columns.
    forward : bool
        True for forward-strand sites, whose partner lies downstream.

    Returns
    -------
    numpy.ndarray
        ``(n_cols,)`` boolean mask of partner columns.
    """
    neighbor = cls.next_idx if forward else cls.prev_idx
    motif = (cls.ca | cls.ta) if forward else (cls.tg | cls.ta2)

    out = np.zeros(cls.arr.shape[1], dtype=bool)
    for i in np.where(site_mask)[0]:
        partners = neighbor[motif[:, i], i]
        partners = partners[partners >= 0]
        if partners.size:
            out[np.bincount(partners).argmax()] = True
    return out


def _label_columns(cls, columns):
    """
    Which alignment positions receive a letter, and which of those are partners.

    Every column is drawn as a bar regardless; this only chooses the lettering.

    Parameters
    ----------
    cls : derip2.aln_ops.ColumnClassification
        Classification of the alignment.
    columns : {'rip', 'substrate', 'all'}
        ``'all'`` letters every position holding a base. ``'rip'`` letters the
        columns with both substrate and product evidence, together with the
        partner base of each detected motif. ``'substrate'`` letters columns
        that hold unmutated substrate but are neither RIP-like, nor corrected,
        nor carrying any deamination product, again with their partners.

    Returns
    -------
    tuple of numpy.ndarray
        ``(label, partner)``, each an ``(n_cols,)`` boolean mask. ``partner``
        is a subset of ``label`` and never overlaps a site column.

    Raises
    ------
    ValueError
        If ``columns`` is not recognised.
    """
    if columns not in COLUMN_SETS:
        raise ValueError(f'columns must be one of {COLUMN_SETS}, got {columns!r}')

    has_bases = cls.base_count > 0
    if columns == 'all':
        return has_bases, np.zeros_like(has_bases)

    rip_like = cls.fwd_col | cls.rev_col
    if columns == 'rip':
        fwd_site, rev_site = cls.fwd_col, cls.rev_col
    else:
        # Substrate that RIP has demonstrably not touched: no product, no
        # non-RIP deamination, and no correction applied to the consensus.
        untouched = (
            ~rip_like
            & ~(cls.modC | cls.modG)
            & ~(cls.nonrip_fwd.any(axis=0) | cls.nonrip_rev.any(axis=0))
        )
        fwd_site = cls.sub_fwd.any(axis=0) & untouched
        rev_site = cls.sub_rev.any(axis=0) & untouched

    site = (fwd_site | rev_site) & has_bases
    partner = (
        _partner_columns(cls, fwd_site, forward=True)
        | _partner_columns(cls, rev_site, forward=False)
    ) & has_bases
    partner &= ~site
    return site | partner, partner


def _strand_direction(cls):
    """
    Per-column strand orientation: +1 forward, -1 reverse, 0 tied.

    Because every non-gap base falls in exactly one of the C/T and G/A pairs,
    the two proportions sum to one and the sign of their difference determines
    the strand. A tie means neither strand holds the majority, and neither
    strand's correction fires.

    Parameters
    ----------
    cls : derip2.aln_ops.ColumnClassification
        Classification of the alignment.

    Returns
    -------
    numpy.ndarray
        ``(n_cols,)`` int array of -1, 0 or +1.
    """
    return np.sign((cls.nC + cls.nT) - (cls.nA + cls.nG)).astype(int)


def _denominator(cls, scale):
    """
    Per-column divisor that turns counts into the plotted bar height.

    Parameters
    ----------
    cls : derip2.aln_ops.ColumnClassification
        Classification of the alignment.
    scale : {'column', 'alignment', 'counts'}
        ``'column'`` normalises each bar to its own non-gap depth, so a full bar
        reaches 1.0. ``'alignment'`` divides by the number of sequences, so a
        column that is 80% gaps reaches only 0.2. ``'counts'`` leaves raw counts.

    Returns
    -------
    numpy.ndarray
        ``(n_cols,)`` float array of divisors, never zero.

    Raises
    ------
    ValueError
        If ``scale`` is not recognised.
    """
    n_rows, n_cols = cls.arr.shape
    if scale == 'column':
        denom = cls.base_count.astype(float)
    elif scale == 'alignment':
        denom = np.full(n_cols, float(n_rows))
    elif scale == 'counts':
        denom = np.ones(n_cols)
    else:
        raise ValueError(f'scale must be one of {SCALES}, got {scale!r}')
    return np.where(denom > 0, denom, 1.0)


def _bar_path(x, y0, width, height, direction, radius):
    """
    Rectangle with its two data-end corners rounded, anchored at the baseline.

    Parameters
    ----------
    x : float
        Bar centre.
    y0 : float
        Baseline-side edge of the segment.
    width, height : float
        Segment size in data coordinates.
    direction : int
        ``+1`` if the segment grows upward, ``-1`` downward.
    radius : float
        Corner radius in data (y) units; clamped so it can never exceed half the
        bar width or the segment height.

    Returns
    -------
    matplotlib.path.Path
        Closed path for the segment.
    """
    from matplotlib.path import Path

    half = width / 2.0
    r = max(0.0, min(radius, half, abs(height)))
    left, right = x - half, x + half
    y1 = y0 + direction * height  # the data end
    inner = y1 - direction * r

    if r <= 0:
        verts = [(left, y0), (left, y1), (right, y1), (right, y0), (left, y0)]
        codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
        return Path(verts, codes)

    verts = [
        (left, y0),
        (left, inner),
        (left, y1),  # control
        (left + r, y1),
        (right - r, y1),
        (right, y1),  # control
        (right, inner),
        (right, y0),
        (left, y0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


def _stack_column(ax, x, y0, segments, bar_width, direction, radius, edge_width):
    """
    Draw one column's stacked segments outward from the gutter edge.

    Segments are separated by a hairline of the surface colour rather than by a
    border, so adjacent fills read as distinct without adding visual weight.
    Only the outermost segment gets rounded corners, so the stack reads as one
    bar with a rounded data-end.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x : float
        Bar centre.
    y0 : float
        Inner edge of the bar (the gutter edge, not necessarily zero).
    segments : list of tuple
        ``(height, colour, label, alpha)`` in stacking order, innermost first.
    bar_width : float
        Bar width in data coordinates.
    direction : int
        ``+1`` to stack upward, ``-1`` to stack downward.
    radius : float
        Corner radius for the outermost segment, in data units.
    edge_width : float
        Width of the surface-coloured separator drawn between segments.

    Returns
    -------
    float
        Total height drawn.
    """
    from matplotlib.patches import PathPatch

    drawn = [s for s in segments if s[0] > 0]
    cursor = y0
    for i, (height, color, _label, alpha) in enumerate(drawn):
        is_outer = i == len(drawn) - 1
        path = _bar_path(
            x, cursor, bar_width, height, direction, radius if is_outer else 0.0
        )
        ax.add_patch(
            PathPatch(
                path,
                facecolor=color,
                edgecolor=SURFACE,
                linewidth=edge_width,
                joinstyle='round',
                alpha=alpha,
                zorder=2,
            )
        )
        cursor += direction * height
    return abs(cursor - y0)


def _nice_ticks(extent, scale):
    """
    Positive tick positions covering ``0..extent`` with a readable step.

    Parameters
    ----------
    extent : float
        Largest bar height on either arm.
    scale : str
        The scale mode; ``'counts'`` gets integer steps.

    Returns
    -------
    numpy.ndarray
        Ascending tick positions, starting at 0 and always reaching at least
        ``extent``.
    """
    if extent <= 0:
        return np.array([0.0])

    if scale == 'counts':
        step = float(max(1, int(np.ceil(extent / 4.0))))
    else:
        for step in (0.1, 0.2, 0.25, 0.5, 1.0):
            if extent / step <= 5:
                break

    # Round the tick count up so the top tick never falls short of the tallest
    # bar, which would leave the data end floating above the last gridline.
    n_steps = int(np.ceil(extent / step - 1e-9))
    return np.arange(n_steps + 1) * step


def plot_strand_bias(
    cls,
    outfile=None,
    mode='rip',
    scale='column',
    stack='signal',
    xaxis='none',
    color_by='base',
    columns='all',
    column_range=None,
    consensus_seq=None,
    title=None,
    width=None,
    height=4.2,
    dpi=300,
    max_columns=None,
    emphasis=True,
    ax=None,
):
    """
    Draw a diverging stacked-bar chart of per-column RIP strand bias.

    Parameters
    ----------
    cls : derip2.aln_ops.ColumnClassification
        Classification produced by :func:`derip2.aln_ops.classify_columns`.
    outfile : str, optional
        Path to write the figure to; format inferred from the extension. Use
        ``.svg`` or ``.pdf`` for publication. If None, nothing is written.
    mode : {'rip', 'non_rip', 'all_deamination'}, optional
        Which deamination events to display (default: ``'rip'``).
    scale : {'column', 'alignment', 'counts'}, optional
        Bar height normalisation (default: ``'column'``).
    stack : {'signal', 'product', 'all'}, optional
        Which bases the bar is made of (default: ``'signal'``). ``'signal'``
        stacks the RIP product and its unmutated substrate; ``'product'`` draws
        the product alone; ``'all'`` adds every remaining base as a translucent
        noise segment. The bar is *never* rescaled, so the missing height
        honestly shows how much of the column was excluded.
    xaxis : {'none', 'logo', 'derip'}, optional
        Decoration drawn in the gutter around the zero line: nothing, a sequence
        logo, or the deRIP'd consensus base (default: ``'none'``). When set, the
        bars are offset from the zero line so the lettering is never obscured.
    color_by : {'base', 'role'}, optional
        Colour segments by nucleotide identity or by the role the base plays
        (default: ``'base'``).
    columns : {'rip', 'substrate', 'all'}, optional
        Which positions are lettered when ``xaxis`` is ``'logo'`` or
        ``'derip'`` (default: ``'all'``). Every column is drawn as a bar
        whatever this is set to. ``'rip'`` letters the RIP-like columns and the
        partner base of each motif; ``'substrate'`` letters only untouched
        substrate columns and their partners.
    column_range : tuple of int, optional
        ``(start, end)`` half-open alignment column range to restrict the plot
        to. Use this to zoom into a region of a large alignment.
    consensus_seq : str, optional
        Gapped deRIP'd consensus, required when ``xaxis='derip'``.
    title : str, optional
        Figure title. Defaults to a description of the mode.
    width : float, optional
        Figure width in inches. Defaults to a width scaled to the column count,
        with no upper bound short of the matplotlib canvas limit.
    height : float, optional
        Figure height in inches (default: 4.2).
    dpi : int, optional
        Raster resolution (default: 300).
    max_columns : int, optional
        Refuse to draw more than this many bars. Unset by default: long
        alignments are drawn in full, on the assumption that the output is
        vector and can be zoomed.
    emphasis : bool, optional
        Wash the columns in which the current mode observed a transition, and
        fade the bars and letters of the columns that merely provide context
        (default: True).
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes instead of creating a figure.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the chart.

    Raises
    ------
    ValueError
        If an option is unrecognised, if ``xaxis='derip'`` without a consensus,
        or if ``max_columns`` is set and more columns would be drawn.

    Notes
    -----
    Bars above the axis are forward-strand columns; bars below are
    reverse-strand columns. Columns where neither strand holds a majority carry
    no correction and are marked with a hatched band rather than dropped
    silently.

    Examples
    --------
    >>> from derip2.aln_ops import classify_alignment
    >>> cls = classify_alignment(alignment)               # doctest: +SKIP
    >>> plot_strand_bias(cls, outfile='bias.svg')         # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    if xaxis not in XAXIS_STYLES:
        raise ValueError(f'xaxis must be one of {XAXIS_STYLES}, got {xaxis!r}')
    if color_by not in COLOR_BY:
        raise ValueError(f'color_by must be one of {COLOR_BY}, got {color_by!r}')
    if stack not in STACK_SETS:
        raise ValueError(f'stack must be one of {STACK_SETS}, got {stack!r}')
    if xaxis == 'derip' and not consensus_seq:
        raise ValueError("xaxis='derip' requires consensus_seq")

    label_mask, partner_mask = _label_columns(cls, columns)

    keep = cls.base_count > 0
    if column_range is not None:
        start, end = column_range
        window = np.zeros_like(keep)
        window[start:end] = True
        keep = keep & window
        label_mask = label_mask & window

    counts = _column_segments(cls, mode)
    denom = _denominator(cls, scale)
    direction = _strand_direction(cls)
    col_idx = np.where(keep)[0]

    # A column the current mode found a transition in. Everything else is
    # context: it is drawn, but it does not compete for attention.
    mutated = (counts['fwd_product'] + counts['rev_product']) > 0

    if col_idx.size == 0:
        logger.warning('No columns to plot for mode=%r', mode)
    if max_columns is not None and col_idx.size > max_columns:
        raise ValueError(
            f'{col_idx.size} columns selected but max_columns={max_columns}. '
            'Restrict with column_range, or raise max_columns.'
        )

    gutter = GUTTER[xaxis]

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        if ax is None:
            width = width or _figure_width(col_idx.size)
            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        else:
            fig = ax.figure

        ax.set_facecolor(SURFACE)
        fig.patch.set_facecolor(SURFACE)

        bar_width = 0.76
        label_colors = {}
        max_extent = 0.0
        highlighted = False
        tied = False

        for x in col_idx:
            d = direction[x]
            is_mutated = bool(mutated[x])

            if emphasis and is_mutated:
                ax.axvspan(
                    x - 0.5,
                    x + 0.5,
                    facecolor=HIGHLIGHT,
                    alpha=HIGHLIGHT_ALPHA,
                    linewidth=0,
                    zorder=0,
                )
                highlighted = True

            if d == 0:
                # Neither strand holds a majority, so neither correction fires
                # and there is no bar to draw. Hatch the column rather than drop
                # it silently. The band sits behind the data, so it leaves the
                # lettering between the two arms untouched.
                ax.axvspan(
                    x - 0.5,
                    x + 0.5,
                    facecolor='none',
                    hatch=TIE_HATCH,
                    edgecolor=GRIDLINE,
                    linewidth=0,
                    zorder=0,
                )
                tied = True
                continue

            if d > 0:
                product = int(counts['fwd_product'][x])
                substrate = int(counts['fwd_substrate'][x])
                p_base, s_base = 'T', 'C'
            else:
                product = int(counts['rev_product'][x])
                substrate = int(counts['rev_substrate'][x])
                p_base, s_base = 'A', 'G'

            other = max(0, int(cls.base_count[x]) - product - substrate)

            if color_by == 'base':
                p_color, s_color = BASE_COLORS[p_base], BASE_COLORS[s_base]
                p_label, s_label = f'{p_base} (product)', f'{s_base} (substrate)'
            else:
                p_color, s_color = ROLE_COLORS['product'], ROLE_COLORS['substrate']
                p_label, s_label = 'RIP product', 'RIP substrate'

            # A context column recedes wholesale; within a mutated bar the noise
            # segment recedes on its own account. Taking the weaker of the two
            # rather than compounding keeps a faded noise segment legible as a
            # bar instead of dissolving into the column wash behind it.
            fade = CONTEXT_BAR_ALPHA if emphasis and not is_mutated else 1.0

            # The product sits against the gutter, so every product segment
            # shares a baseline and RIP extent can be compared across columns.
            segments = [(product / denom[x], p_color, p_label, fade)]
            if stack != 'product':
                segments.append((substrate / denom[x], s_color, s_label, fade))
            if stack == 'all':
                segments.append(
                    (
                        other / denom[x],
                        ROLE_COLORS['other'],
                        'Other bases',
                        min(NOISE_ALPHA, fade),
                    )
                )

            drawn = _stack_column(
                ax,
                x,
                d * gutter,
                segments,
                bar_width,
                d,
                radius=0.012,
                edge_width=0.7,
            )
            max_extent = max(max_extent, drawn)
            for _, color, label, _alpha in segments:
                label_colors[label] = color

        _draw_gutter(
            ax,
            cls,
            label_mask,
            partner_mask,
            mutated if emphasis else None,
            xaxis,
            gutter,
            consensus_seq,
            bar_width,
        )
        _finish(
            ax,
            fig,
            col_idx,
            label_colors,
            title,
            mode,
            scale,
            gutter,
            max_extent,
            color_by,
            highlighted,
            tied,
        )

        if outfile:
            _warn_if_subpixel(outfile, fig, col_idx.size, dpi)
            fig.savefig(outfile, dpi=dpi, bbox_inches='tight', facecolor=SURFACE)
            logger.info('Strand bias figure saved to %s', outfile)

    return fig


def _figure_width(n_cols):
    """
    Figure width in inches for a given column count, clamped to the canvas limit.

    Parameters
    ----------
    n_cols : int
        Number of bars to be drawn.

    Returns
    -------
    float
        Width in inches.
    """
    width = max(6.0, 0.17 * max(n_cols, 14) + 2.4)
    if width > MAX_FIG_INCHES:
        logger.warning(
            'Figure width for %d columns clamped to %g inches. Bars will be '
            'narrow; write to .svg or .pdf and zoom.',
            n_cols,
            MAX_FIG_INCHES,
        )
        width = MAX_FIG_INCHES
    return width


def _warn_if_subpixel(outfile, fig, n_cols, dpi):
    """
    Warn when a raster target cannot resolve the bars it is about to be given.

    Parameters
    ----------
    outfile : str
        Destination path; the extension decides whether the output is raster.
    fig : matplotlib.figure.Figure
        The figure about to be written.
    n_cols : int
        Number of bars drawn.
    dpi : int
        Raster resolution the figure will be written at.

    Returns
    -------
    None
        Nothing is returned; a warning is logged if the bars are sub-pixel.
    """
    if not str(outfile).lower().endswith(RASTER_EXTS):
        return
    pixels = fig.get_size_inches()[0] * dpi
    if n_cols > pixels / 2.0:
        logger.warning(
            '%d columns across %d pixels: bars are sub-pixel and the raster '
            'figure will misrepresent them. Write to .svg or .pdf instead.',
            n_cols,
            int(pixels),
        )


def _letter_alpha(is_partner, is_mutated):
    """
    Opacity for one gutter letter.

    Parameters
    ----------
    is_partner : bool
        The letter is the context half of a dinucleotide, not the site itself.
    is_mutated : bool or None
        The column carries a transition under the current mode. None when
        emphasis is off, in which case every site letter is drawn at full
        strength.

    Returns
    -------
    float
        Opacity in ``(0, 1]``.
    """
    if is_partner:
        return PARTNER_ALPHA
    if is_mutated is None or is_mutated:
        return SITE_ALPHA
    return CONTEXT_ALPHA


def _draw_gutter(
    ax, cls, label_mask, partner_mask, mutated, xaxis, gutter, consensus_seq, bar_width
):
    """
    Draw the sequence logo or the deRIP'd consensus letters in the gutter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    cls : derip2.aln_ops.ColumnClassification
        Classification of the alignment.
    label_mask : numpy.ndarray
        ``(n_cols,)`` boolean mask of the positions to letter.
    partner_mask : numpy.ndarray
        ``(n_cols,)`` boolean mask of the lettered positions that are the context
        half of a dinucleotide rather than the deaminated site.
    mutated : numpy.ndarray or None
        ``(n_cols,)`` boolean mask of the columns the current mode found a
        transition in, or None when emphasis is off.
    xaxis : {'none', 'logo', 'derip'}
        Which decoration to draw. ``'none'`` draws nothing.
    gutter : float
        Half-height of the lettering band, in y-units.
    consensus_seq : str or None
        Gapped deRIP'd consensus, required when ``xaxis='derip'``.
    bar_width : float
        Bar width in data coordinates; the glyphs are sized from it.

    Returns
    -------
    None
        Nothing is returned; the letters are added to ``ax``.
    """
    if xaxis == 'none':
        return

    from derip2.plotting.logo import draw_gap, draw_glyph, draw_logo_column

    # Letters fill the gutter band on both sides of the zero line. A partner
    # letter is drawn shorter as well as fainter, so a motif reads as a bold
    # site followed by a receding context base.
    band = 2.0 * gutter * 0.92
    glyph_width = bar_width * 0.86

    for x in np.where(label_mask)[0]:
        is_partner = bool(partner_mask[x])
        is_mutated = None if mutated is None else bool(mutated[x])
        alpha = _letter_alpha(is_partner, is_mutated)
        height = band * (PARTNER_SCALE if is_partner else 1.0)

        if xaxis == 'logo':
            draw_logo_column(
                ax,
                cls.base_counts[x, :4],  # A, C, G, T
                x,
                -height / 2.0,
                height,
                BASE_COLORS,
                bar_width=glyph_width,
                alpha=alpha,
            )
            continue

        # 'derip'. The column still holds bases, and so still carries a bar, but
        # the deRIP'd consensus really is gapped here: say so with a dash.
        base = consensus_seq[x].upper()
        if base not in BASE_COLORS:
            draw_gap(
                ax,
                x,
                0.0,
                glyph_width * 0.5,
                band * 0.16,
                INK_MUTED,
                alpha=alpha,
            )
            continue

        draw_glyph(
            ax,
            base,
            x - glyph_width / 2.0,
            -height / 2.0,
            glyph_width,
            height,
            BASE_COLORS[base],
            alpha=alpha,
        )


def _legend_order(color_by):
    """
    Fixed legend order: forward pair, reverse pair, then the noise slot.

    Parameters
    ----------
    color_by : {'base', 'role'}
        Whether segments are coloured by nucleotide identity or by role.

    Returns
    -------
    list of str
        Segment labels in the order they should appear in the legend. Labels
        that were never drawn are filtered out by the caller.
    """
    if color_by == 'base':
        return [
            'T (product)',
            'C (substrate)',
            'A (product)',
            'G (substrate)',
            'Other bases',
        ]
    return ['RIP product', 'RIP substrate', 'Other bases']


def _annotate_arms(ax):
    """
    Name the two arms inside the panel.

    A reader should never have to infer that "below the axis" means "reverse
    strand". The labels are inset from the axes corners by a fixed number of
    *points*, not by a fraction of the axes, so they stay next to the y-axis
    however wide the alignment makes the figure.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.

    Returns
    -------
    None
        Nothing is returned; the annotations are added to ``ax``.
    """
    for xy, offset, va, text in (
        ((0, 1), (4, -4), 'top', 'forward strand  CA → TA'),
        ((0, 0), (4, 4), 'bottom', 'reverse strand  TG → TA'),
    ):
        ax.annotate(
            text,
            xy=xy,
            xycoords='axes fraction',
            xytext=offset,
            textcoords='offset points',
            ha='left',
            va=va,
            fontsize=ANNOTATION_SIZE,
            color=INK_SECONDARY,
        )


def _finish(
    ax,
    fig,
    col_idx,
    label_colors,
    title,
    mode,
    scale,
    gutter,
    extent,
    color_by,
    highlighted=False,
    tied=False,
):
    """
    Apply chrome: baselines, spines, ticks, strand annotations and legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to decorate.
    fig : matplotlib.figure.Figure
        The figure ``ax`` belongs to.
    col_idx : numpy.ndarray
        Ascending indices of the columns that were drawn, used to set the x-limits.
    label_colors : dict
        Mapping of segment label to colour, for the legend.
    title : str or None
        Figure title. Falls back to a description of ``mode``.
    mode : {'rip', 'non_rip', 'all_deamination'}
        The display mode, used for the default title.
    scale : {'column', 'alignment', 'counts'}
        The height normalisation, used for the y-label and tick steps.
    gutter : float
        Half-height of the lettering band, in y-units.
    extent : float
        Tallest bar drawn on either arm.
    color_by : {'base', 'role'}
        Whether segments are coloured by nucleotide identity or by role.
    highlighted : bool, optional
        A mutated-column wash was drawn, so the legend needs its key
        (default: False).
    tied : bool, optional
        A tie hatch was drawn, so the legend needs its key (default: False).

    Returns
    -------
    None
        Nothing is returned; the chrome is added to ``ax`` and ``fig``.
    """
    from matplotlib.patches import Patch

    if col_idx.size:
        pad = max(1.0, 0.02 * (col_idx.max() - col_idx.min() + 1))
        ax.set_xlim(col_idx.min() - pad, col_idx.max() + pad)

    ticks = _nice_ticks(extent, scale)
    limit = gutter + max(ticks.max(), extent) * ARM_LABEL_HEADROOM + 0.02

    # Ticks are mirrored about the gutter: the sign of a bar is its strand, so
    # labelling the lower arm "-0.5" would read as a negative proportion.
    # Zero is dropped from the arms and re-added once per baseline, so a gutter
    # does not produce three stacked "0" labels.
    arm = ticks[ticks > 0]
    positions = np.concatenate([-arm[::-1] - gutter, [-gutter, gutter], arm + gutter])
    magnitudes = np.concatenate([arm[::-1], [0.0, 0.0], arm])
    if gutter == 0:
        positions = np.concatenate([-arm[::-1], [0.0], arm])
        magnitudes = np.concatenate([arm[::-1], [0.0], arm])

    ax.set_yticks(positions)
    ax.set_yticklabels([f'{m:g}' for m in magnitudes])
    ax.set_ylim(-limit, limit)

    # Bars rest on the gutter edges, so that is where the baselines belong. With
    # no gutter the two collapse onto the zero line. Drawing a rule at zero when
    # a gutter exists would strike through the lettering.
    for y in {0.0} if gutter == 0 else {-gutter, gutter}:
        ax.axhline(y, color=BASELINE, linewidth=0.6, zorder=1)

    y_label = {
        'column': 'Proportion of column',
        'alignment': 'Proportion of sequences',
        'counts': 'Sequences',
    }[scale]
    ax.set_ylabel(y_label, color=INK_SECONDARY, fontsize=AXIS_LABEL_SIZE)
    ax.set_xlabel('Alignment column', color=INK_SECONDARY, fontsize=AXIS_LABEL_SIZE)
    ax.set_title(
        title or f'{MODE_TITLES[mode]}: strand bias',
        color=INK_PRIMARY,
        fontsize=TITLE_SIZE,
        pad=8,
        loc='center',
    )

    # No gridlines: the two baselines and the y-ticks carry the scale, and a
    # grid would rule through the region washes.
    ax.grid(False)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(AXIS_INK)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(
        colors=AXIS_INK,
        labelcolor=INK_MUTED,
        labelsize=TICK_LABEL_SIZE,
        direction='out',
        length=2.5,
        width=0.6,
    )

    _annotate_arms(ax)

    ordered = [lab for lab in _legend_order(color_by) if lab in label_colors]
    handles = [
        Patch(facecolor=label_colors[lab], edgecolor=SURFACE, linewidth=0.6, label=lab)
        for lab in ordered
    ]
    if highlighted:
        handles.append(
            Patch(
                facecolor=HIGHLIGHT,
                alpha=HIGHLIGHT_ALPHA,
                edgecolor=SURFACE,
                linewidth=0.6,
                label='Mutated column',
            )
        )
    if tied:
        handles.append(
            Patch(
                facecolor='none',
                hatch=TIE_HATCH,
                edgecolor=GRIDLINE,
                linewidth=0.6,
                label='No strand majority',
            )
        )

    # A column wash needs a key even when no bar was drawn to carry one.
    if handles:
        ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.14),
            ncol=len(handles),
            frameon=False,
            fontsize=LEGEND_SIZE,
            labelcolor=INK_SECONDARY,
            handlelength=1.0,
            handleheight=1.0,
            columnspacing=1.4,
        )

    fig.tight_layout()
