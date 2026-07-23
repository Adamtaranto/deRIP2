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

# Opacity of subject bases that match the deRIP'd reference; mismatches are 1.0.
MATCH_ALPHA = 0.3


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


def _gene_exon_path(x0, x1, y0, y1, strand, rx, ry, arrow_len):
    """
    Build a rounded rectangle with an arrowhead at the strand's 3' end.

    The exon body runs from ``x0`` to ``x1`` between ``y0`` and ``y1``. The end
    in the transcription direction (right for ``'+'``, left for ``'-'``) tapers
    to a point; the opposite two corners are rounded. Separate ``rx``/``ry`` radii
    are used because the annotation axes is very wide and short, so the x and y
    data scales differ by orders of magnitude.

    Parameters
    ----------
    x0, x1 : float
        Left and right column bounds of the exon (x0 < x1).
    y0, y1 : float
        Top and bottom band bounds (y0 < y1).
    strand : str
        ``'+'`` (arrow right) or ``'-'`` (arrow left).
    rx, ry : float
        Corner rounding radius in x (columns) and y (band units).
    arrow_len : float
        Length of the arrowhead along x, in columns.

    Returns
    -------
    matplotlib.path.Path
        A closed path for the exon glyph.
    """
    from matplotlib.path import Path

    ymid = (y0 + y1) / 2.0
    w = x1 - x0
    a = max(0.0, min(arrow_len, w * 0.5))
    rx = max(0.0, min(rx, (w - a) * 0.5))
    ry = max(0.0, min(ry, (y1 - y0) * 0.5))

    if strand == '-':
        base = x0 + a  # arrow base x
        verts = [
            (x1 - rx, y0),
            (base, y0),
            (x0, ymid),  # arrow tip
            (base, y1),
            (x1 - rx, y1),
            (x1, y1),
            (x1, y1 - ry),  # bottom-right round
            (x1, y0 + ry),
            (x1, y0),
            (x1 - rx, y0),  # top-right round
            (x1 - rx, y0),
        ]
    else:
        base = x1 - a
        verts = [
            (x0 + rx, y0),
            (base, y0),
            (x1, ymid),  # arrow tip
            (base, y1),
            (x0 + rx, y1),
            (x0, y1),
            (x0, y1 - ry),  # bottom-left round
            (x0, y0 + ry),
            (x0, y0),
            (x0 + rx, y0),  # top-left round
            (x0 + rx, y0),
        ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


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

        from matplotlib.collections import PathCollection

        bar_width = 0.82
        radius = 0.18
        # Build every bar's rounded path and collect them into a SINGLE artist.
        # A heavily-RIP'd sequence contributes ~1000 bars; adding them one patch
        # at a time triggers a per-patch limit update and bloats the SVG, so a
        # PathCollection (one add, one <g>) is far faster to draw and serialise.
        paths, facecolors = [], []
        for cols, direction, color in (
            (above_prod, +1, PRODUCT_COLOR),
            (above_sub, +1, SUBSTRATE_COLOR),
            (below_prod, -1, PRODUCT_COLOR),
            (below_sub, -1, SUBSTRATE_COLOR),
        ):
            for x in cols:
                paths.append(_bar_path(x, 0.0, bar_width, 1.0, direction, radius))
                facecolors.append(color)
        if paths:
            ax.add_collection(
                PathCollection(
                    paths,
                    facecolors=facecolors,
                    edgecolors=SURFACE,
                    linewidths=0.4,
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
    cds_tracks=None,
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
    cds_tracks : list of tuple, optional
        Gene-annotation tracks to draw in a sub-plot below the alignment rows,
        each ``(exon_spans, strand, stop_columns, label, colour)``: the per-exon
        ``(start_col, end_col)`` spans (drawn as rounded segments with an
        arrowhead at the ``strand`` 3' end and joined across introns by a
        midline), this subject's stop-codon columns (marked ``*`` above the
        track), a row label and a hex colour.
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
        alpha : float or numpy.ndarray
            Opacity for every non-gap base; either a scalar or a ``(n_cols,)``
            per-column array (gaps stay opaque white regardless).

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

    # Subject bases that match the deRIP'd reference are drawn faded so the eye
    # is drawn to the mismatches (where RIP or other change acted); mismatches
    # stay at full opacity. Gaps are handled inside _base_rgba (kept opaque).
    cons_bytes = (
        np.frombuffer(consensus_seq.upper().encode('ascii'), dtype='S1')
        if consensus_seq is not None
        else None
    )
    if cons_bytes is not None:
        subj_alpha = np.where(row == cons_bytes, MATCH_ALPHA, 1.0)
    else:
        subj_alpha = 1.0
    subject_rgba = _base_rgba(row, subj_alpha)[np.newaxis, :, :]

    # Columns the whole-MSA strand-bias plot would shade: a RIP product observed
    # on either strand anywhere in the column (same rule as plot_strand_bias).
    mutated = np.where((cls.prod_fwd.sum(axis=0) + cls.prod_rev.sum(axis=0)) > 0)[0]

    # This sequence's own role at each column, for the triangle markers.
    marker_sets = (
        (np.where(cls.prod_fwd[i] | cls.prod_rev[i])[0], PRODUCT_COLOR),
        (np.where(cls.sub_fwd[i] | cls.sub_rev[i])[0], SUBSTRATE_COLOR),
        (np.where(cls.nonrip_fwd[i] | cls.nonrip_rev[i])[0], NONRIP_COLOR),
    )

    tracks = list(cds_tracks or [])
    n_tracks = len(tracks)

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig_w = width or _figure_width(n_cols)
        if n_tracks:
            # Annotations live in their own sub-plot below the alignment rows,
            # sharing the column axis. The main plot keeps a fixed ~2 in; the
            # track plot grows with the number of gene tracks.
            fig, (ax, ax_ann) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(fig_w, height),
                gridspec_kw={
                    'height_ratios': [2.0, max(0.6, 0.75 * n_tracks)],
                    'hspace': 0.32,
                },
            )
            ax_ann.set_facecolor(SURFACE)
        else:
            fig, ax = plt.subplots(figsize=(fig_w, height))
            ax_ann = None
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        # --- main plot: subject + deRIP rows, markers, RIP-column shading -------
        ax.imshow(
            subject_rgba,
            aspect='auto',
            interpolation='nearest',
            extent=(-0.5, n_cols - 0.5, 1.0, 0.0),
            zorder=2,
        )
        yticks, row_labels = [0.5], [seq_id or f'row {i}']
        ref_bottom = 2.0 + GAP
        if cons_bytes is not None:
            ax.imshow(
                _base_rgba(cons_bytes, 1.0)[np.newaxis, :, :],
                aspect='auto',
                interpolation='nearest',
                extent=(-0.5, n_cols - 0.5, ref_bottom, 1.0 + GAP),
                zorder=2,
            )
            yticks.append(1.5 + GAP)
            row_labels.append('deRIP')
        else:
            ref_bottom = 1.0

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
        y_top, y_bottom = MARKER_Y - 0.35, ref_bottom + bottom_pad
        ax.set_ylim(y_bottom, y_top)  # inverted

        # RIP-like column shading as a single raster layer (one artist, one SVG
        # element) rather than one axvspan per mutated column.
        if mutated.size:
            highlight = np.zeros((1, n_cols, 4), dtype=float)
            highlight[0, mutated, :3] = _hex_to_rgb(HIGHLIGHT)
            highlight[0, mutated, 3] = HIGHLIGHT_ALPHA
            ax.imshow(
                highlight,
                aspect='auto',
                interpolation='nearest',
                extent=(-0.5, n_cols - 0.5, y_bottom, y_top),
                zorder=0,
            )

        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(row_labels, fontsize=TICK_LABEL_SIZE, color=INK_MUTED)
        ax.tick_params(
            colors=AXIS_INK, labelsize=TICK_LABEL_SIZE, length=2.5, width=0.6
        )
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS_INK)
        ax.spines['bottom'].set_linewidth(0.6)

        # --- annotation sub-plot: gene models -----------------------------------
        x_axis_ax = ax
        fig.annotation_titles = {}  # gid -> tooltip text, injected as SVG <title>
        if ax_ann is not None:
            x_axis_ax = ax_ann
            fig.annotation_titles = _draw_annotation_tracks(
                ax_ann, tracks, n_cols, fig_w
            )
            ax.tick_params(labelbottom=False)  # x labels belong to the track axis

        x_axis_ax.set_xlabel(
            'Alignment column', color=INK_SECONDARY, fontsize=AXIS_LABEL_SIZE
        )

        # Untitled by default; the HTML report supplies its own section heading.
        if title:
            ax.set_title(
                title, color=INK_PRIMARY, fontsize=TITLE_SIZE, pad=6, loc='center'
            )

        if outfile is not None:
            fig.savefig(outfile, dpi=dpi, bbox_inches='tight', facecolor=SURFACE)
            logger.info('Sequence row strip saved to %s', outfile)

    return fig


def _draw_annotation_tracks(ax, tracks, n_cols, fig_w, show_labels=True):
    """
    Draw gene-model annotation tracks into a dedicated sub-plot axis.

    Each gene is one row: its exons are rounded segments with an arrowhead at the
    strand's 3' end, joined across introns by a midline of the gene colour, and a
    bold red ``*`` marks each stop codon just above the track.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The annotation sub-plot axis (shares the column x-axis with the rows).
    tracks : list of tuple
        ``(exon_spans, strand, stop_columns, label, colour[, cds_id])`` per gene.
        ``label`` is the y-axis row label (the parent transcript / gene id); the
        optional 6th element ``cds_id`` is the CDS ``ID`` attribute used for the
        hover tooltip so distinct CDS isoforms sharing one transcript stay
        distinguishable. Legacy 5-tuples fall back to ``label`` for the tooltip.
    n_cols : int
        Number of alignment columns (for the shared x-range).
    fig_w : float
        Figure width in inches, used to scale the arrowhead length in columns.
    show_labels : bool, optional
        Whether to label each track row with its gene id on the y-axis. The
        per-sequence strips label them; the alignment-wide plot suppresses the
        labels and relies on hover tooltips instead (default: True).

    Returns
    -------
    dict of str to str
        Maps each exon patch's ``gid`` to the tooltip text (the annotation id,
        plus the CDS exon number) that the report injects as a ``data-tip``.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch

    ax.set_facecolor(SURFACE)
    half_h = 0.22
    # Arrowhead length in columns: a small, roughly font-sized fraction of the
    # visible width so it reads at report scale without swamping short exons.
    arrow_len = max(1.0, n_cols / max(fig_w, 1.0) * 0.06)
    rx = arrow_len * 0.7

    # Each exon is a separate patch (not a collection) so it can carry a gid and,
    # after SVG export, an individual <title> tooltip. Exon counts are small.
    titles = {}
    gid_n = 0
    for t, track in enumerate(tracks):
        exon_spans, strand, stop_cols, label, colour = track[:5]
        # The hover tooltip identifies the CDS by its own ``ID`` (6th element);
        # the y-axis row label keeps the parent transcript id. Legacy 5-tuples
        # fall back to the transcript id for the tooltip too.
        tip_id = track[5] if len(track) > 5 else label
        center = t + 0.55
        y0, y1 = center - half_h, center + half_h
        spans = [(float(s), float(e)) for s, e in exon_spans]
        if spans:
            gmin = min(s for s, _ in spans)
            gmax = max(e for _, e in spans)
            # Join exons with a midline behind the segments (intron line).
            ax.add_line(
                Line2D(
                    [gmin - 0.5, gmax + 0.5],
                    [center, center],
                    color=colour,
                    linewidth=1.1,
                    zorder=1,
                )
            )
            n_exons = len(spans)
            for j, (s, e) in enumerate(spans):
                # Number exons in transcription order (5'->3'): ascending for the
                # plus strand, descending for the minus strand.
                exon_no = n_exons - j if strand == '-' else j + 1
                gid = f'anntip{gid_n}'
                gid_n += 1
                titles[gid] = f'{tip_id} — CDS exon {exon_no}/{n_exons}'
                patch = PathPatch(
                    _gene_exon_path(
                        s - 0.5, e + 0.5, y0, y1, strand, rx, half_h * 0.6, arrow_len
                    ),
                    facecolor=colour,
                    edgecolor=SURFACE,
                    linewidth=0.4,
                    zorder=2,
                )
                patch.set_gid(gid)
                ax.add_patch(patch)
        for sc in stop_cols:
            ax.text(
                int(sc),
                center - half_h - 0.16,
                '*',
                ha='center',
                va='center',
                fontsize=9,
                fontweight='bold',
                color='#b4292a',
                zorder=4,
            )

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(len(tracks) + 0.05, -0.05)  # inverted, room for '*' above track 0
    if show_labels:
        ax.set_yticks([t + 0.55 for t in range(len(tracks))])
        ax.set_yticklabels(
            [track[3] for track in tracks],
            fontsize=TICK_LABEL_SIZE,
            color=INK_MUTED,
        )
    else:
        ax.set_yticks([])
    ax.tick_params(colors=AXIS_INK, labelsize=TICK_LABEL_SIZE, length=2.5, width=0.6)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(AXIS_INK)
    ax.spines['bottom'].set_linewidth(0.6)
    return titles


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
