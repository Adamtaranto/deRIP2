"""
Native matplotlib figures for flanking-context spectra of RIP-like sites.

Each spectrum is a 16-bar histogram of the ``[up][center][down]`` flank context
(the two centre bases fixed, the two flanks varying). The six spectra for one
sequence — substrate and product site states, each as combined / forward /
reverse strand views — are drawn as a single 2x3 grid so the whole set embeds as
one figure (one SVG id-prefix in the per-sequence report).

The palette, typography and light publication surface are shared with the rest of
the package: the substrate/product bar colours come from
:mod:`derip2.plotting.persequence` (blue substrate, orange product, matching the
per-sequence strand-bias strip), and the axis styling, motif-tick emphasis and
save helpers are reused from :mod:`derip2.plotting.spectra`.
"""

import logging
from typing import List, Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from derip2.plotting.persequence import PRODUCT_COLOR, SUBSTRATE_COLOR
from derip2.plotting.spectra import (
    CONTEXT_TICK_SIZE,
    _save,
    _style_axes,
)
from derip2.plotting.strandbias import (
    AXIS_LABEL_SIZE,
    FONT_STACK,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    TITLE_SIZE,
)
from derip2.spectra.flank_channels import (
    FLANK16_LABELS_CA,
    FLANK16_LABELS_TA,
)

logger = logging.getLogger(__name__)

# The 2x3 grid layout: rows are site states, columns are strand views.
_STATES = ('substrate', 'product')
_STRANDS = ('combined', 'forward', 'reverse')
_STRAND_TITLES = {
    'combined': 'Combined',
    'forward': 'Forward',
    'reverse': 'Reverse',
}
_STATE_LABELS = {
    'substrate': 'Substrate (CpA)',
    'product': 'Product (TpA)',
}
_STATE_COLORS = {
    'substrate': SUBSTRATE_COLOR,
    'product': PRODUCT_COLOR,
}
_STATE_LABELS_LIST = {
    'substrate': FLANK16_LABELS_CA,
    'product': FLANK16_LABELS_TA,
}

# Short caption distinguishing this context model in figure headings.
_FLANK_CAPTION = (
    r'flank context of RIP-like sites (5$^\prime$-N[XY]N-3$^\prime$; '
    r'centre XY fixed, flanks vary)'
)


def _draw_flank_ticks(ax, x: np.ndarray, motifs: List[str]) -> None:
    """
    Draw monospace 4 bp motif tick labels with the two centre bases bolded.

    Mirrors :func:`derip2.plotting.spectra._draw_motif_ticks` but bolds the two
    fixed centre bases (indices 1 and 2) instead of a single mutated base. Each
    label is drawn as two overlaid real-monospace layers — a regular-weight layer
    carrying the flanks and a bold layer carrying the centre — so the emphasis
    stays column-aligned and every 4 bp label keeps the same width.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to label.
    x : numpy.ndarray
        The bar x positions, one per motif.
    motifs : list of str
        The 4-character motifs in bar order (e.g. ``'ACAA'``).

    Returns
    -------
    None
        The labels are drawn in place.
    """
    bold_positions = {1, 2}

    def _blank(motif: str, keep_center: bool) -> str:
        # Keep either the centre dinucleotide (bold layer) or the two flanks
        # (regular layer), blanking the rest so both layers stay aligned.
        return ''.join(
            ch if (i in bold_positions) == keep_center else ' '
            for i, ch in enumerate(motif)
        )

    flanks = [_blank(m, keep_center=False) for m in motifs]
    ax.set_xticks(x)
    labels = ax.set_xticklabels(
        flanks,
        rotation=90,
        fontsize=CONTEXT_TICK_SIZE,
        family='monospace',
        color=INK_SECONDARY,
    )
    for label, motif in zip(labels, motifs):
        overlay = ax.text(0, 0, _blank(motif, keep_center=True))
        overlay.update_from(label)
        overlay.set_position(label.get_position())
        overlay.set_transform(label.get_transform())
        overlay.set_fontweight('bold')
        overlay.set_color(INK_PRIMARY)


def _draw_flank_panel(
    ax,
    counts: np.ndarray,
    state: str,
    *,
    percentage: bool = False,
    show_ticks: bool = True,
) -> None:
    """
    Draw one 16-bar flank-context panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw into.
    counts : numpy.ndarray
        ``(16,)`` channel counts in canonical flank order.
    state : {'substrate', 'product'}
        Which site state, selecting the bar colour and centre-dinucleotide label.
    percentage : bool, optional
        Rescale the bars so they sum to 100 (default: raw counts).
    show_ticks : bool, optional
        Draw the per-bar motif tick labels (default: True).

    Returns
    -------
    None
        The panel is drawn in place.
    """
    values = counts.astype(float)
    if percentage:
        total = values.sum()
        if total > 0:
            values = 100.0 * values / total

    x = np.arange(16)
    ax.bar(x, values, width=0.8, color=_STATE_COLORS[state], linewidth=0)
    _style_axes(ax)
    ax.set_xlim(-0.75, 15.75)
    ax.margins(x=0)
    if show_ticks:
        _draw_flank_ticks(ax, x, list(_STATE_LABELS_LIST[state]))
    else:
        ax.set_xticks([])


def _draw_grid(fig, axes, vectors, *, percentage: bool) -> None:
    """
    Populate a 2x3 axes grid with the six state x strand flank panels.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The parent figure (styled to the shared surface).
    axes : numpy.ndarray
        ``(2, 3)`` array of axes: rows are :data:`_STATES`, columns
        :data:`_STRANDS`.
    vectors : dict
        Maps ``(state, strand)`` to its ``(16,)`` count vector.
    percentage : bool
        Passed through to each panel.

    Returns
    -------
    None
        The grid is drawn in place.
    """
    fig.patch.set_facecolor(SURFACE)
    for r, state in enumerate(_STATES):
        for c, strand in enumerate(_STRANDS):
            ax = axes[r, c]
            _draw_flank_panel(
                ax, vectors[(state, strand)], state, percentage=percentage
            )
            if r == 0:
                ax.set_title(
                    _STRAND_TITLES[strand],
                    fontsize=AXIS_LABEL_SIZE,
                    color=INK_PRIMARY,
                )
            if c == 0:
                ax.set_ylabel(
                    _STATE_LABELS[state],
                    fontsize=AXIS_LABEL_SIZE,
                    color=INK_PRIMARY,
                )


def plot_flank_spectra_grid(
    result,
    sample: int,
    outfile: Optional[str] = None,
    *,
    title: Optional[str] = None,
    percentage: bool = False,
    width: float = 11.0,
    panel_height: float = 2.0,
    dpi: int = 300,
    bare: bool = False,
):
    """
    Draw the six flank-context spectra for one sequence as a 2x3 grid.

    Rows are the two site states (substrate, product); columns are the three
    strand views (combined, forward, reverse). Every panel is a 16-bar flank
    histogram sharing the package palette and motif-tick styling.

    Parameters
    ----------
    result : derip2.stats.flank_spectra.FlankSpectraResult
        The computed spectra.
    sample : int
        Column index into ``result.sample_names`` (the alignment row to draw).
    outfile : str or None, optional
        Output path; when ``None`` the figure is returned unsaved.
    title : str or None, optional
        Figure heading (omitted when ``bare``).
    percentage : bool, optional
        Plot each panel as a percentage of its own total (default: counts).
    width : float, optional
        Figure width in inches (default: 11.0), wide enough for the motif ticks.
    panel_height : float, optional
        Height per grid row in inches (default: 2.0).
    dpi : int, optional
        Raster resolution (default: 300).
    bare : bool, optional
        Omit the caption/suptitle for embedding under an external heading
        (default: False).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.

    Raises
    ------
    IndexError
        If ``sample`` is out of range for the available samples.
    """
    n_samples = len(result.sample_names)
    if not -n_samples <= sample < n_samples:
        raise IndexError(f'sample {sample} out of range for {n_samples} sample(s)')
    s = sample % n_samples
    vectors = {
        (state, strand): result.matrix(state, strand)[:, s]
        for state in _STATES
        for strand in _STRANDS
    }

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, axes = plt.subplots(2, 3, figsize=(width, panel_height * 2), squeeze=False)
        _draw_grid(fig, axes, vectors, percentage=percentage)
        if not bare:
            heading = f'{title}\n{_FLANK_CAPTION}' if title else _FLANK_CAPTION
            fig.suptitle(heading, fontsize=TITLE_SIZE, color=INK_PRIMARY)
        fig.tight_layout()
        _save(fig, outfile, dpi)
    return fig


def plot_flank_spectra_pooled(
    result,
    outfile: Optional[str] = None,
    *,
    title: Optional[str] = None,
    percentage: bool = False,
    width: float = 11.0,
    panel_height: float = 2.0,
    dpi: int = 300,
    bare: bool = False,
):
    """
    Draw the six flank-context spectra pooled across every sequence.

    Same 2x3 layout as :func:`plot_flank_spectra_grid`, but on the
    alignment-wide row-summed counts (:meth:`FlankSpectraResult.pooled`), for the
    report's overview page.

    Parameters
    ----------
    result : derip2.stats.flank_spectra.FlankSpectraResult
        The computed spectra.
    outfile : str or None, optional
        Output path; when ``None`` the figure is returned unsaved.
    title : str or None, optional
        Figure heading (omitted when ``bare``).
    percentage : bool, optional
        Plot each panel as a percentage of its own total (default: counts).
    width : float, optional
        Figure width in inches (default: 11.0).
    panel_height : float, optional
        Height per grid row in inches (default: 2.0).
    dpi : int, optional
        Raster resolution (default: 300).
    bare : bool, optional
        Omit the caption/suptitle for embedding (default: False).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    pooled = result.pooled()
    # Precompute combined vectors so the closure stays simple.
    vectors = {
        ('substrate', 'forward'): pooled['sub_fwd'],
        ('substrate', 'reverse'): pooled['sub_rev'],
        ('substrate', 'combined'): pooled['sub_fwd'] + pooled['sub_rev'],
        ('product', 'forward'): pooled['prod_fwd'],
        ('product', 'reverse'): pooled['prod_rev'],
        ('product', 'combined'): pooled['prod_fwd'] + pooled['prod_rev'],
    }

    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, axes = plt.subplots(2, 3, figsize=(width, panel_height * 2), squeeze=False)
        _draw_grid(fig, axes, vectors, percentage=percentage)
        if not bare:
            heading = f'{title}\n{_FLANK_CAPTION}' if title else _FLANK_CAPTION
            fig.suptitle(heading, fontsize=TITLE_SIZE, color=INK_PRIMARY)
        fig.tight_layout()
        _save(fig, outfile, dpi)
    return fig
