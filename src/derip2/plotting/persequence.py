"""
Single-sequence figures for the per-sequence HTML report.

These render *one* alignment row at a time, in the same visual system as the
alignment-wide strand-bias chart (:mod:`derip2.plotting.strandbias`): the same
colourblind-validated palette, typography and light publication surface, so the
per-sequence report reads as one document with the rest of the package.

Two figures are provided:

- :func:`per_sequence_strand_bias` — a fixed-height binary bar strip showing, for
  one sequence, which RIP-like columns carry a forward-strand event (above the
  axis) or a reverse-strand event (below), coloured by whether the base is the
  RIP product or the surviving substrate.
- :func:`sequence_row_strip` — a coloured strip of the single alignment row with
  RIP/deamination sites highlighted.

Both return the matplotlib figure so the report can embed it as inline SVG.
"""

import logging
from typing import Optional

import matplotlib

matplotlib.use('Agg')  # non-interactive backend for headless report generation
import matplotlib.pyplot as plt
import numpy as np

from derip2.plotting.strandbias import (
    ANNOTATION_SIZE,
    AXIS_INK,
    AXIS_LABEL_SIZE,
    BASELINE,
    FONT_STACK,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    LEGEND_SIZE,
    ROLE_COLORS,
    SURFACE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    _bar_path,
    _figure_width,
)

logger = logging.getLogger(__name__)

# Grey used for the base fills of the alignment-row strip: a neutral backdrop so
# the RIP highlight overlay carries all the colour. Matches the 'basegrey'
# palette in :mod:`derip2.plotting.minialign`.
STRIP_BASE = '#c7d1d0'
STRIP_GAP = '#ffffff'

# Per-sequence role palette. The report deliberately swaps the alignment-wide
# convention so that in these figures the RIP *product* is orange and the
# surviving *substrate* is blue; kept as named constants so the alignment row,
# strand-bias strip and RIP-completion bar all agree.
PRODUCT_COLOR = ROLE_COLORS['substrate']  # orange: the RIP product base (T / A)
SUBSTRATE_COLOR = ROLE_COLORS['product']  # blue: the surviving substrate

# Highlight colours for the alignment-row strip, drawn over the grey bases, plus
# a third hue for non-RIP deamination (only shown when reaminate is on).
HIGHLIGHT_COLORS = {
    'product': PRODUCT_COLOR,  # orange
    'substrate': SUBSTRATE_COLOR,  # blue
    'non_rip': '#a05fb4',  # violet: deamination outside RIP context
}


def _hex_to_rgb(color: str):
    """
    Convert a ``#rrggbb`` hex string to a float RGB triple in ``[0, 1]``.

    Parameters
    ----------
    color : str
        Hex colour, e.g. ``'#2a78d6'``.

    Returns
    -------
    tuple of float
        ``(r, g, b)`` each in ``[0, 1]``.
    """
    color = color.lstrip('#')
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def per_sequence_strand_bias(
    cls,
    row_index: int,
    *,
    seq_id: Optional[str] = None,
    title: Optional[str] = None,
    height: float = 2.2,
    width: Optional[float] = None,
    dpi: int = 300,
    outfile: Optional[str] = None,
    ax=None,
):
    """
    Draw a fixed-height binary strand-bias strip for a single sequence.

    Every RIP-like column that this sequence participates in becomes one bar of
    unit height. A forward-strand event is drawn above the axis, a reverse-strand
    event below it, and the bar is coloured by the role the sequence's base
    plays: the RIP *product* (the deaminated base) or the surviving *substrate*.

    Because a cell holds one base, each column contributes at most one bar: a
    forward product (T of TpA), a forward substrate (C of CpA), a reverse product
    (A of TpA) or a reverse substrate (G of TpG). The four are mutually exclusive
    per cell, so the strip is unambiguous.

    Parameters
    ----------
    cls : derip2.aln_ops.ColumnClassification
        Classification of the whole alignment; only row ``row_index`` is read.
    row_index : int
        Index of the sequence (alignment row) to plot.
    seq_id : str, optional
        Sequence identifier, used in the default title.
    title : str, optional
        Figure title. Defaults to a description naming ``seq_id``.
    height : float, optional
        Figure height in inches (default: 2.2).
    width : float, optional
        Figure width in inches. Defaults to a width scaled to the number of
        columns (:func:`derip2.plotting.strandbias._figure_width`); pass an
        explicit value to control bar density for a scrolling container.
    dpi : int, optional
        Raster resolution when ``outfile`` is a raster path (default: 300).
    outfile : str, optional
        Path to write the figure to. When ``None`` the figure is returned unsaved.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. When given, no new figure is created and
        ``outfile`` is ignored.

    Returns
    -------
    matplotlib.figure.Figure
        The figure the strip was drawn on.
    """
    i = row_index
    n_cols = cls.arr.shape[1]

    # Row-sliced cell masks. Each is a (n_cols,) boolean; the four are mutually
    # exclusive within a column because they require different bases.
    above_prod = np.where(cls.prod_fwd[i])[0]  # T, RIP'd (TA), forward
    above_sub = np.where(cls.sub_fwd[i])[0]  # C, un-RIP'd (CA), forward
    below_prod = np.where(cls.prod_rev[i])[0]  # A, RIP'd (TA), reverse
    below_sub = np.where(cls.sub_rev[i])[0]  # G, substrate (TG), reverse

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        if ax is None:
            fig, ax = plt.subplots(figsize=(width or _figure_width(n_cols), height))
        else:
            fig = ax.figure

        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        from matplotlib.patches import PathPatch

        bar_width = 0.82
        radius = 0.18
        # (columns, direction, colour) -> one call to the rounded-bar helper each.
        for cols, direction, color in (
            (above_prod, +1, PRODUCT_COLOR),
            (above_sub, +1, SUBSTRATE_COLOR),
            (below_prod, -1, PRODUCT_COLOR),
            (below_sub, -1, SUBSTRATE_COLOR),
        ):
            for x in cols:
                path = _bar_path(x, 0.0, bar_width, 1.0, direction, radius)
                ax.add_patch(
                    PathPatch(
                        path,
                        facecolor=color,
                        edgecolor=SURFACE,
                        linewidth=0.4,
                        joinstyle='round',
                        zorder=2,
                    )
                )

        # Chrome: baseline, limits, spines, ticks.
        ax.axhline(0.0, color=BASELINE, linewidth=0.6, zorder=1)
        pad = max(1.0, 0.02 * n_cols)
        ax.set_xlim(-pad, n_cols - 1 + pad)
        ax.set_ylim(-1.35, 1.35)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['1', '0', '1'])
        ax.set_ylabel('RIP-like site', color=INK_SECONDARY, fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel('Alignment column', color=INK_SECONDARY, fontsize=AXIS_LABEL_SIZE)

        # A title is only drawn when explicitly requested; the HTML report names
        # each section with its own heading, so the default is untitled.
        if title:
            ax.set_title(
                title, color=INK_PRIMARY, fontsize=TITLE_SIZE, pad=8, loc='center'
            )

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

        _annotate_binary_arms(ax)

        from matplotlib.patches import Patch

        handles = [
            Patch(
                facecolor=PRODUCT_COLOR,
                edgecolor=SURFACE,
                linewidth=0.6,
                label='RIP product (TA)',
            ),
            Patch(
                facecolor=SUBSTRATE_COLOR,
                edgecolor=SURFACE,
                linewidth=0.6,
                label='substrate (CA / TG)',
            ),
        ]
        ax.legend(
            handles=handles,
            loc='upper right',
            fontsize=LEGEND_SIZE,
            frameon=False,
            ncol=2,
        )

        if outfile is not None and ax is fig.axes[0]:
            fig.savefig(outfile, dpi=dpi, bbox_inches='tight', facecolor=SURFACE)
            logger.info('Per-sequence strand bias figure saved to %s', outfile)

    return fig


def _annotate_binary_arms(ax):
    """
    Name the forward/reverse arms of a per-sequence strand-bias strip.

    Mirrors :func:`derip2.plotting.strandbias._annotate_arms` but for the
    fixed-height binary strip, whose arms carry the same meaning.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.

    Returns
    -------
    None
        The annotations are added to ``ax``.
    """
    for xy, offset, va, text in (
        ((0, 1), (4, -4), 'top', 'forward strand  CA / TA'),
        ((0, 0), (4, 4), 'bottom', 'reverse strand  TG / TA'),
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


def sequence_row_strip(
    cls,
    row_index: int,
    *,
    seq_id: Optional[str] = None,
    consensus_seq: Optional[str] = None,
    title: Optional[str] = None,
    height: float = 1.0,
    width: Optional[float] = None,
    dpi: int = 300,
    outfile: Optional[str] = None,
):
    """
    Draw one alignment row as a coloured strip with RIP sites highlighted.

    The row's bases are drawn as a neutral grey band (gaps white); RIP products,
    surviving substrates and — when ``reaminate`` was on — non-RIP deaminations
    are then overlaid in the role palette, so a reader sees at a glance where and
    how RIP acted on this sequence.

    Parameters
    ----------
    cls : derip2.aln_ops.ColumnClassification
        Classification of the whole alignment; only row ``row_index`` is read.
    row_index : int
        Index of the sequence (alignment row) to draw.
    seq_id : str, optional
        Sequence identifier, used in the default title.
    consensus_seq : str, optional
        The deRIP'd consensus (one base per column). When given, a second row is
        drawn beneath the sequence showing the reconstructed base.
    title : str, optional
        Figure title. Defaults to a description naming ``seq_id``.
    height : float, optional
        Figure height in inches (default: 1.0).
    width : float, optional
        Figure width in inches. Defaults to a width scaled to the number of
        columns; pass an explicit value to match the strand-bias strip so the
        two line up column-for-column in a scrolling container.
    dpi : int, optional
        Raster resolution when ``outfile`` is a raster path (default: 300).
    outfile : str, optional
        Path to write the figure to. When ``None`` the figure is returned unsaved.

    Returns
    -------
    matplotlib.figure.Figure
        The figure the strip was drawn on.
    """
    i = row_index
    row = cls.arr[i]
    n_cols = row.shape[0]
    n_rows = 2 if consensus_seq is not None else 1

    # Base layer: grey for a base, white for a gap.
    base_rgb = np.empty((n_rows, n_cols, 3), dtype=float)
    is_gap = row == b'-'
    base_rgb[0, :] = _hex_to_rgb(STRIP_BASE)
    base_rgb[0, is_gap] = _hex_to_rgb(STRIP_GAP)
    if consensus_seq is not None:
        cons = np.frombuffer(consensus_seq.upper().encode('ascii'), dtype='S1')
        cons_gap = cons == b'-'
        base_rgb[1, :] = _hex_to_rgb(STRIP_BASE)
        base_rgb[1, cons_gap] = _hex_to_rgb(STRIP_GAP)

    # Overlay: paint RIP roles onto the sequence row (row 0 of the image).
    overlays = (
        (cls.prod_fwd[i] | cls.prod_rev[i], 'product'),
        (cls.sub_fwd[i] | cls.sub_rev[i], 'substrate'),
        (cls.nonrip_fwd[i] | cls.nonrip_rev[i], 'non_rip'),
    )
    for mask, role in overlays:
        cols = np.where(mask)[0]
        if cols.size:
            base_rgb[0, cols] = _hex_to_rgb(HIGHLIGHT_COLORS[role])

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, ax = plt.subplots(figsize=(width or _figure_width(n_cols), height))
        fig.patch.set_facecolor(SURFACE)
        ax.imshow(base_rgb, aspect='auto', interpolation='nearest')

        row_labels = [seq_id or f'row {i}']
        if consensus_seq is not None:
            row_labels.append('deRIP')
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=TICK_LABEL_SIZE, color=INK_MUTED)
        ax.set_xlabel('Alignment column', color=INK_SECONDARY, fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(
            colors=AXIS_INK, labelsize=TICK_LABEL_SIZE, length=2.5, width=0.6
        )
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS_INK)
        ax.spines['bottom'].set_linewidth(0.6)

        # Untitled by default; the HTML report supplies its own section heading.
        if title:
            ax.set_title(
                title, color=INK_PRIMARY, fontsize=TITLE_SIZE, pad=6, loc='center'
            )

        if outfile is not None:
            fig.savefig(outfile, dpi=dpi, bbox_inches='tight', facecolor=SURFACE)
            logger.info('Sequence row strip saved to %s', outfile)

    return fig


def rip_completion_bar(
    stats,
    *,
    title: Optional[str] = None,
    width: float = 6.6,
    height: float = 1.7,
    dpi: int = 300,
    outfile: Optional[str] = None,
):
    """
    Draw horizontal stacked bars of the fraction of RIP-like sites that are RIP'd.

    For each strand, the available RIP-like sites are the surviving substrate
    dinucleotides plus the RIP products (converted sites). The bar shows what
    fraction of that substrate has been converted — the RIP *product* segment —
    against the intact substrate, so a nearly full blue bar is a heavily RIP'd
    strand and a nearly empty one has escaped RIP.

    Three bars are drawn: forward, reverse and the two combined.

    Parameters
    ----------
    stats : Mapping
        A per-sequence statistics row exposing ``fwd_product``,
        ``fwd_substrate``, ``rev_product`` and ``rev_substrate`` (as produced by
        :meth:`derip2.derip.DeRIP.summarize_stats`).
    title : str, optional
        Figure title.
    width, height : float, optional
        Figure size in inches. Fixed by default so the bars line up across the
        per-sequence report's pages.
    dpi : int, optional
        Raster resolution when ``outfile`` is a raster path (default: 300).
    outfile : str, optional
        Path to write the figure to. When ``None`` the figure is returned unsaved.

    Returns
    -------
    matplotlib.figure.Figure
        The figure the bars were drawn on.
    """
    fwd_p = float(stats['fwd_product'])
    fwd_s = float(stats['fwd_substrate'])
    rev_p = float(stats['rev_product'])
    rev_s = float(stats['rev_substrate'])

    # (label, product, substrate), top to bottom.
    rows = [
        ('Forward (CA)', fwd_p, fwd_s),
        ('Reverse (TG)', rev_p, rev_s),
        ('Combined', fwd_p + rev_p, fwd_s + rev_s),
    ]

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, ax = plt.subplots(figsize=(width, height))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        y = np.arange(len(rows))[::-1]  # first row at the top
        bar_h = 0.6
        for yi, (_label, product, substrate) in zip(y, rows):
            total = product + substrate
            if total <= 0:
                # No RIP-like sites on this strand: draw an empty track.
                ax.text(
                    50,
                    yi,
                    'no RIP-like sites',
                    ha='center',
                    va='center',
                    fontsize=ANNOTATION_SIZE,
                    color=INK_MUTED,
                )
                pct = 0.0
            else:
                pct = 100.0 * product / total
                ax.barh(
                    yi,
                    pct,
                    height=bar_h,
                    color=PRODUCT_COLOR,
                    edgecolor=SURFACE,
                    linewidth=0.4,
                    zorder=2,
                )
                ax.barh(
                    yi,
                    100.0 - pct,
                    left=pct,
                    height=bar_h,
                    color=SUBSTRATE_COLOR,
                    edgecolor=SURFACE,
                    linewidth=0.4,
                    zorder=2,
                )
            ax.text(
                101,
                yi,
                f'{pct:.0f}%',
                ha='left',
                va='center',
                fontsize=TICK_LABEL_SIZE,
                color=INK_SECONDARY,
            )

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([label for label, _p, _s in rows], fontsize=TICK_LABEL_SIZE)
        ax.set_xlabel(
            'RIP-like sites converted (%)',
            color=INK_SECONDARY,
            fontsize=AXIS_LABEL_SIZE,
        )
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(
            colors=AXIS_INK,
            labelcolor=INK_MUTED,
            labelsize=TICK_LABEL_SIZE,
            length=2.5,
            width=0.6,
        )
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS_INK)
        ax.spines['bottom'].set_linewidth(0.6)

        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(
                    facecolor=PRODUCT_COLOR,
                    edgecolor=SURFACE,
                    linewidth=0.6,
                    label="RIP'd (product)",
                ),
                Patch(
                    facecolor=SUBSTRATE_COLOR,
                    edgecolor=SURFACE,
                    linewidth=0.6,
                    label='intact (substrate)',
                ),
            ],
            loc='lower center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            fontsize=LEGEND_SIZE,
            frameon=False,
        )

        if title:
            ax.set_title(
                title, color=INK_PRIMARY, fontsize=TITLE_SIZE, pad=22, loc='center'
            )

        # Fixed margins so the plot body is identical on every page.
        fig.subplots_adjust(left=0.18, right=0.9, top=0.78, bottom=0.24)

        if outfile is not None:
            fig.savefig(outfile, dpi=dpi, facecolor=SURFACE)
            logger.info('RIP completion bar saved to %s', outfile)

    return fig


# GC-content bar palette: a single hue split for the two base-pair classes.
GC_COLOR = '#3a8fb7'  # teal for G+C
AT_COLOR = '#c7d1d0'  # grey for A+T


def gc_content_bar(
    stats,
    *,
    title: Optional[str] = None,
    width: float = 6.6,
    height: float = 1.0,
    dpi: int = 300,
    outfile: Optional[str] = None,
):
    """
    Draw a horizontal stacked bar of this sequence's GC content.

    The bar is split into the G+C fraction (filled) and the A+T remainder, with
    the percentage labelled. RIP lowers GC by converting C to T, so a low bar is
    consistent with heavy RIP.

    Parameters
    ----------
    stats : Mapping
        A per-sequence statistics row exposing ``GC`` (a fraction in ``[0, 1]``),
        as produced by :meth:`derip2.derip.DeRIP.summarize_stats`.
    title : str, optional
        Figure title.
    width, height : float, optional
        Figure size in inches. Fixed by default so the bar lines up across pages.
    dpi : int, optional
        Raster resolution when ``outfile`` is a raster path (default: 300).
    outfile : str, optional
        Path to write the figure to. When ``None`` the figure is returned unsaved.

    Returns
    -------
    matplotlib.figure.Figure
        The figure the bar was drawn on.
    """
    gc_pct = 100.0 * float(stats['GC'])
    gc_pct = min(100.0, max(0.0, gc_pct))

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, ax = plt.subplots(figsize=(width, height))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        ax.barh(
            0,
            gc_pct,
            height=0.6,
            color=GC_COLOR,
            edgecolor=SURFACE,
            linewidth=0.4,
            zorder=2,
        )
        ax.barh(
            0,
            100.0 - gc_pct,
            left=gc_pct,
            height=0.6,
            color=AT_COLOR,
            edgecolor=SURFACE,
            linewidth=0.4,
            zorder=2,
        )
        ax.text(
            101,
            0,
            f'{gc_pct:.0f}%',
            ha='left',
            va='center',
            fontsize=TICK_LABEL_SIZE,
            color=INK_SECONDARY,
        )

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([0])
        ax.set_yticklabels(['GC'], fontsize=TICK_LABEL_SIZE)
        ax.set_xlabel(
            'Base composition (%)', color=INK_SECONDARY, fontsize=AXIS_LABEL_SIZE
        )
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(
            colors=AXIS_INK,
            labelcolor=INK_MUTED,
            labelsize=TICK_LABEL_SIZE,
            length=2.5,
            width=0.6,
        )
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS_INK)
        ax.spines['bottom'].set_linewidth(0.6)

        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(
                    facecolor=GC_COLOR, edgecolor=SURFACE, linewidth=0.6, label='G+C'
                ),
                Patch(
                    facecolor=AT_COLOR, edgecolor=SURFACE, linewidth=0.6, label='A+T'
                ),
            ],
            loc='lower center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            fontsize=LEGEND_SIZE,
            frameon=False,
        )

        if title:
            ax.set_title(
                title, color=INK_PRIMARY, fontsize=TITLE_SIZE, pad=22, loc='center'
            )

        fig.subplots_adjust(left=0.12, right=0.9, top=0.66, bottom=0.42)

        if outfile is not None:
            fig.savefig(outfile, dpi=dpi, facecolor=SURFACE)
            logger.info('GC content bar saved to %s', outfile)

    return fig
