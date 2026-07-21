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
- :func:`sequence_row_strip` — the subject and deRIP'd reference rows drawn
  base-by-base, with RIP-like columns shaded as in the alignment-wide plot.
- :func:`rip_completion_bar` / :func:`gc_content_bar` — small horizontal
  stacked-bar summaries.

All return the matplotlib figure so the report can embed it as inline SVG.
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
    BASE_COLORS,
    BASELINE,
    FONT_STACK,
    HIGHLIGHT,
    HIGHLIGHT_ALPHA,
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
NONRIP_COLOR = '#a05fb4'  # violet: deamination outside RIP dinucleotide context


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
    Draw the subject and deRIP'd reference rows as base-coloured strips.

    Both aligned sequences are drawn base-by-base, coloured by nucleotide
    identity with the shared palette (A green, C blue, G violet, T red; gaps
    white). The subject is drawn on top and the reconstructed deRIP'd reference
    below it, separated by a narrow gap. Triangle markers above the subject flag
    its role at each column, and columns the whole-alignment strand-bias analysis
    marks as RIP-like — those carrying a RIP product on either strand anywhere in
    the alignment — are shaded with the same hueless wash used by
    :func:`derip2.plotting.strandbias.plot_strand_bias`.

    Parameters
    ----------
    cls : derip2.aln_ops.ColumnClassification
        Classification of the whole alignment; row ``row_index`` is the subject.
    row_index : int
        Index of the subject sequence (alignment row) to draw.
    seq_id : str, optional
        Subject sequence identifier, used to label its row.
    consensus_seq : str, optional
        The deRIP'd reference (one base per column), drawn as the second row.
    title : str, optional
        Figure title. Untitled by default (the report supplies a heading).
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

    def _base_rgba(bytes_row, alpha):
        """
        Colour one sequence row by base identity into an RGBA image row.

        Parameters
        ----------
        bytes_row : numpy.ndarray
            ``(n_cols,)`` ``'S1'`` bases for the row.
        alpha : float
            Opacity applied to every non-gap base (gaps stay opaque white).

        Returns
        -------
        numpy.ndarray
            ``(n_cols, 4)`` RGBA array.
        """
        rgba = np.empty((n_cols, 4), dtype=float)
        rgba[:, 3] = alpha
        # Default any non-ACGT (N, IUPAC) to the neutral grey.
        rgba[:, :3] = _hex_to_rgb(STRIP_BASE)
        for base, hexcol in BASE_COLORS.items():
            rgba[bytes_row == base.encode('ascii'), :3] = _hex_to_rgb(hexcol)
        gap = bytes_row == b'-'
        rgba[gap, :3] = _hex_to_rgb(STRIP_GAP)
        rgba[gap, 3] = 1.0  # gaps stay opaque white on both rows
        return rgba

    # y-layout (data coords, axis inverted so smaller y is higher on the page):
    #   markers  at  y = MARKER_Y  (above the subject row)
    #   subject  in  [0, 1]
    #   gap      in  [1, 1 + GAP]  (narrow whitespace)
    #   deRIP    in  [1 + GAP, 2 + GAP]
    GAP = 0.22
    MARKER_Y = -0.5
    subject_rgba = _base_rgba(row, 1.0)[np.newaxis, :, :]

    # Columns the whole-MSA strand-bias plot would shade: a RIP product observed
    # on either strand anywhere in the column (same rule as plot_strand_bias).
    mutated = np.where((cls.prod_fwd.sum(axis=0) + cls.prod_rev.sum(axis=0)) > 0)[0]

    # This sequence's own role at each column, for the triangle markers.
    marker_sets = (
        (np.where(cls.prod_fwd[i] | cls.prod_rev[i])[0], PRODUCT_COLOR),
        (np.where(cls.sub_fwd[i] | cls.sub_rev[i])[0], SUBSTRATE_COLOR),
        (np.where(cls.nonrip_fwd[i] | cls.nonrip_rev[i])[0], NONRIP_COLOR),
    )

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, ax = plt.subplots(figsize=(width or _figure_width(n_cols), height))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        ax.imshow(
            subject_rgba,
            aspect='auto',
            interpolation='nearest',
            extent=(-0.5, n_cols - 0.5, 1.0, 0.0),
            zorder=2,
        )
        yticks, row_labels = [0.5], [seq_id or f'row {i}']
        ref_bottom = 2.0 + GAP
        if consensus_seq is not None:
            cons = np.frombuffer(consensus_seq.upper().encode('ascii'), dtype='S1')
            ax.imshow(
                _base_rgba(cons, 1.0)[np.newaxis, :, :],
                aspect='auto',
                interpolation='nearest',
                extent=(-0.5, n_cols - 0.5, ref_bottom, 1.0 + GAP),
                zorder=2,
            )
            yticks.append(1.5 + GAP)
            row_labels.append('deRIP')
        else:
            ref_bottom = 1.0

        # Triangle markers above the subject row, coloured by this sequence's role.
        for cols, colour in marker_sets:
            if cols.size:
                ax.scatter(
                    cols,
                    np.full(cols.shape, MARKER_Y),
                    marker='v',
                    s=14,
                    c=colour,
                    edgecolors='none',
                    zorder=3,
                    clip_on=False,
                )

        bottom_pad = 0.35
        ax.set_ylim(ref_bottom + bottom_pad, MARKER_Y - 0.35)  # inverted
        for x in mutated:
            ax.axvspan(
                x - 0.5,
                x + 0.5,
                facecolor=HIGHLIGHT,
                alpha=HIGHLIGHT_ALPHA,
                linewidth=0,
                zorder=0,
            )

        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_yticks(yticks)
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
        A per-sequence statistics row exposing ``GC`` (a percentage in
        ``[0, 100]``), as produced by :meth:`derip2.derip.DeRIP.summarize_stats`.
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
    # ``GC`` is already a percentage in [0, 100] (see summarize_stats).
    gc_pct = min(100.0, max(0.0, float(stats['GC'])))

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
