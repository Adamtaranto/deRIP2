"""
Native matplotlib figures for flanking-context spectra of RIP-like sites.

Each figure is three **bihistograms** — one per strand view (combined, forward,
reverse) — comparing the two site states back to back: surviving **substrate**
counts extend to the **left** and realised RIP **product** counts to the
**right** of a shared centre line, one row per ``[up][centre][down]`` flank
channel. Because a substrate ``CpA`` motif and its product ``TpA`` share the same
flanks, every row is labelled by the **CA-state** motif (e.g. ``GCAG`` labels the
substrate ``GCAG`` and the equivalent product ``GTAG``). Channels whose enrichment
differs significantly between the two states are marked.

The palette, typography and light publication surface are shared with the rest of
the package: the substrate/product bar colours come from
:mod:`derip2.plotting.persequence` (blue substrate, orange product, matching the
per-sequence strand-bias strip), and the axis styling and save helpers are reused
from :mod:`derip2.plotting.spectra`.
"""

import logging
from typing import Optional

import matplotlib

matplotlib.use('Agg')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from derip2.plotting.persequence import PRODUCT_COLOR, SUBSTRATE_COLOR
from derip2.plotting.spectra import (
    CONTEXT_TICK_SIZE,
    _save,
    _style_axes,
)
from derip2.plotting.strandbias import (
    ANNOTATION_SIZE,
    AXIS_LABEL_SIZE,
    FONT_STACK,
    INK_PRIMARY,
    INK_SECONDARY,
    LEGEND_SIZE,
    SURFACE,
    TITLE_SIZE,
)
from derip2.spectra.flank_channels import FLANK16_LABELS_CA, FLANK16_LABELS_TA
from derip2.stats.flank_spectra import differential_channels

logger = logging.getLogger(__name__)

# The three strand views, drawn left-to-right as separate bihistograms.
_STRANDS = ('combined', 'forward', 'reverse')
_STRAND_TITLES = {
    'combined': 'Combined',
    'forward': 'Forward',
    'reverse': 'Reverse',
}

# Accent for the significance markers on differentially enriched motifs (the
# validated deRIP2 C>T red, so it reads as "changed" and stays colourblind-safe).
SIG_COLOR = '#e34948'

# Short caption distinguishing this context model in figure headings.
_FLANK_CAPTION = (
    r'flank context of RIP-like sites (5$^\prime$-N[XY]N-3$^\prime$; '
    r'substrate $\leftarrow$ CA-state | TA-state $\rightarrow$ product)'
)


def _abs_formatter(value, _pos):
    """
    Tick formatter rendering the magnitude of a signed count.

    Parameters
    ----------
    value : float
        The (possibly negative) axis value.
    _pos : int
        Tick position (unused).

    Returns
    -------
    str
        The absolute value formatted without a sign.
    """
    return f'{abs(value):g}'


def _draw_bihistogram(
    ax,
    substrate: np.ndarray,
    product: np.ndarray,
    *,
    sig_mask: np.ndarray,
    show_labels: bool,
    show_right_labels: bool,
    percentage: bool,
) -> None:
    """
    Draw one back-to-back bihistogram of substrate (left) vs product (right).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw into.
    substrate, product : numpy.ndarray
        ``(16,)`` channel counts for the two states, in canonical flank order.
    sig_mask : numpy.ndarray
        ``(16,)`` boolean mask of channels differentially enriched between the
        states; those rows get a significance marker.
    show_labels : bool
        Draw the per-row **CA-state** (substrate) motif tick labels on the left
        y-axis (only the leftmost panel of a multi-panel figure needs them, since
        the rows align).
    show_right_labels : bool
        Draw the per-row **TA-state** (product) motif labels on a right-hand
        y-axis (only the rightmost panel needs them), so each row is named by both
        the substrate motif on the left and its product equivalent on the right.
    percentage : bool
        Rescale each state to sum to 100 before plotting (default is raw counts).

    Returns
    -------
    None
        The panel is drawn in place.
    """
    sub = np.asarray(substrate, dtype=float)
    prod = np.asarray(product, dtype=float)
    if percentage:
        if sub.sum() > 0:
            sub = 100.0 * sub / sub.sum()
        if prod.sum() > 0:
            prod = 100.0 * prod / prod.sum()

    y = np.arange(16)
    # Substrate grows left (negative), product grows right (positive).
    ax.barh(y, -sub, height=0.8, color=SUBSTRATE_COLOR, linewidth=0, zorder=3)
    ax.barh(y, prod, height=0.8, color=PRODUCT_COLOR, linewidth=0, zorder=3)

    _style_axes(ax)
    ax.axvline(0.0, color=INK_PRIMARY, linewidth=0.8, zorder=4)
    span = max(float(sub.max(initial=0.0)), float(prod.max(initial=0.0)), 1.0)
    ax.set_xlim(-span * 1.08, span * 1.08)
    ax.set_ylim(-0.6, 15.6)
    ax.invert_yaxis()  # first channel (ACAA) at the top
    ax.xaxis.set_major_formatter(FuncFormatter(_abs_formatter))

    # Mark the differentially enriched channels with a bold asterisk pinned just
    # inside the right spine. (A full-row highlight is deliberately avoided: with
    # very large pooled counts almost every context is significant, so a per-row
    # band would flood the panel — the effect sizes in the table carry the nuance.)
    trans = ax.get_yaxis_transform()  # x in axes fraction, y in data coords
    for i in np.nonzero(sig_mask)[0]:
        ax.text(
            0.99,
            i,
            '*',
            transform=trans,
            ha='right',
            va='center',
            color=SIG_COLOR,
            fontsize=AXIS_LABEL_SIZE + 2,
            fontweight='bold',
            zorder=5,
        )

    ax.set_yticks(y)
    if show_labels:
        labels = ax.set_yticklabels(
            list(FLANK16_LABELS_CA),
            fontsize=CONTEXT_TICK_SIZE,
            family='monospace',
            color=INK_SECONDARY,
        )
        # Bold + darken the labels of differentially enriched channels.
        for i, label in enumerate(labels):
            if sig_mask[i]:
                label.set_color(INK_PRIMARY)
                label.set_fontweight('bold')
    else:
        ax.set_yticklabels([])

    # Right-hand y-axis: the equivalent TA-state (product) motif for each row,
    # drawn as free text just outside the right spine so it needs no extra axes
    # (keeping the figure at three panels). Significant rows are bolded to match.
    if show_right_labels:
        for i, motif in enumerate(FLANK16_LABELS_TA):
            ax.text(
                1.02,
                i,
                motif,
                transform=trans,
                ha='left',
                va='center',
                family='monospace',
                fontsize=CONTEXT_TICK_SIZE,
                color=INK_PRIMARY if sig_mask[i] else INK_SECONDARY,
                fontweight='bold' if sig_mask[i] else 'normal',
            )


def _strand_vectors(result, sample: Optional[int]):
    """
    Assemble per-strand substrate/product count vectors for a bihistogram figure.

    Parameters
    ----------
    result : derip2.stats.flank_spectra.FlankSpectraResult
        The computed spectra.
    sample : int or None
        Alignment row (sample column) to draw; ``None`` pools every sequence.

    Returns
    -------
    dict
        Maps each strand in :data:`_STRANDS` to ``(substrate, product)`` count
        vectors, each ``(16,)``.
    """
    if sample is None:
        pooled = result.pooled()
        sub = {
            'forward': pooled['sub_fwd'],
            'reverse': pooled['sub_rev'],
            'combined': pooled['sub_fwd'] + pooled['sub_rev'],
        }
        prod = {
            'forward': pooled['prod_fwd'],
            'reverse': pooled['prod_rev'],
            'combined': pooled['prod_fwd'] + pooled['prod_rev'],
        }
    else:
        sub = {
            strand: result.matrix('substrate', strand)[:, sample] for strand in _STRANDS
        }
        prod = {
            strand: result.matrix('product', strand)[:, sample] for strand in _STRANDS
        }
    return {strand: (sub[strand], prod[strand]) for strand in _STRANDS}


def _draw_bihistogram_figure(
    vectors, *, title, percentage, width, panel_height, min_sites, alpha, bare
):
    """
    Render the three-strand bihistogram figure from prepared count vectors.

    Parameters
    ----------
    vectors : dict
        Maps each strand to its ``(substrate, product)`` count vectors.
    title : str or None
        Figure heading (omitted when ``bare``).
    percentage : bool
        Plot each state as a percentage of its own total.
    width : float
        Figure width in inches.
    panel_height : float
        Height of the (single-row) figure in inches.
    min_sites : int
        Minimum sites per state for a channel to be eligible for significance.
    alpha : float
        Per-channel two-sided significance level.
    bare : bool
        Omit the caption/suptitle for embedding under an external heading.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, axes = plt.subplots(1, 3, figsize=(width, panel_height), squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for c, strand in enumerate(_STRANDS):
            ax = axes[0, c]
            sub, prod = vectors[strand]
            sig = differential_channels(sub, prod, min_sites=min_sites, alpha=alpha)
            _draw_bihistogram(
                ax,
                sub,
                prod,
                sig_mask=sig,
                show_labels=(c == 0),
                show_right_labels=(c == len(_STRANDS) - 1),
                percentage=percentage,
            )
            ax.set_title(
                _STRAND_TITLES[strand], fontsize=AXIS_LABEL_SIZE, color=INK_PRIMARY
            )
            ax.set_xlabel(
                '% of state' if percentage else 'sites',
                fontsize=ANNOTATION_SIZE,
                color=INK_SECONDARY,
            )

        # Shared legend: colour = state (direction), plus the significance marker.
        handles = [
            Patch(facecolor=SUBSTRATE_COLOR, label='Substrate (CpA) ←'),
            Patch(facecolor=PRODUCT_COLOR, label='→ Product (TpA)'),
            Line2D(
                [],
                [],
                marker='*',
                markersize=8,
                color=SIG_COLOR,
                linestyle='none',
                label='differentially enriched',
            ),
        ]
        fig.legend(
            handles=handles,
            loc='lower center',
            ncol=3,
            frameon=False,
            fontsize=LEGEND_SIZE,
            bbox_to_anchor=(0.5, -0.02),
        )

        if not bare:
            heading = f'{title}\n{_FLANK_CAPTION}' if title else _FLANK_CAPTION
            fig.suptitle(heading, fontsize=TITLE_SIZE, color=INK_PRIMARY)
        # Reserve a right margin for the free-text TA-state labels (they sit
        # outside the axes, so tight_layout does not account for them) and a
        # bottom strip for the legend.
        fig.tight_layout(rect=(0, 0.04, 0.955, 1))
    return fig


def plot_flank_bihistograms(
    result,
    sample: int,
    outfile: Optional[str] = None,
    *,
    title: Optional[str] = None,
    percentage: bool = False,
    width: float = 11.0,
    panel_height: float = 5.2,
    min_sites: int = 20,
    alpha: float = 0.05,
    dpi: int = 300,
    bare: bool = False,
):
    """
    Draw the three flank-context bihistograms for one sequence.

    One bihistogram per strand view (combined, forward, reverse): substrate
    counts extend left, product counts right, one row per CA-state flank channel.
    Channels differentially enriched between the two states (adjusted
    standardised residual beyond the ``alpha`` critical value, when both states
    have at least ``min_sites`` sites) are highlighted and marked.

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
        Plot each state as a percentage of its own total (default: counts).
    width : float, optional
        Figure width in inches (default: 11.0).
    panel_height : float, optional
        Figure height in inches (default: 5.2), tall enough for 16 rows.
    min_sites : int, optional
        Minimum sites per state for a channel to be eligible for a significance
        mark (default: 20).
    alpha : float, optional
        Per-channel two-sided significance level (default: 0.05).
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
    vectors = _strand_vectors(result, sample % n_samples)
    fig = _draw_bihistogram_figure(
        vectors,
        title=title,
        percentage=percentage,
        width=width,
        panel_height=panel_height,
        min_sites=min_sites,
        alpha=alpha,
        bare=bare,
    )
    _save(fig, outfile, dpi)
    return fig


def plot_flank_bihistograms_pooled(
    result,
    outfile: Optional[str] = None,
    *,
    title: Optional[str] = None,
    percentage: bool = False,
    width: float = 11.0,
    panel_height: float = 5.2,
    min_sites: int = 20,
    alpha: float = 0.05,
    dpi: int = 300,
    bare: bool = False,
):
    """
    Draw the three flank-context bihistograms pooled across every sequence.

    Same layout as :func:`plot_flank_bihistograms`, but on the alignment-wide
    row-summed counts (:meth:`FlankSpectraResult.pooled`), for the report's
    overview page.

    Parameters
    ----------
    result : derip2.stats.flank_spectra.FlankSpectraResult
        The computed spectra.
    outfile : str or None, optional
        Output path; when ``None`` the figure is returned unsaved.
    title : str or None, optional
        Figure heading (omitted when ``bare``).
    percentage : bool, optional
        Plot each state as a percentage of its own total (default: counts).
    width : float, optional
        Figure width in inches (default: 11.0).
    panel_height : float, optional
        Figure height in inches (default: 5.2).
    min_sites : int, optional
        Minimum sites per state for a channel to be eligible for a significance
        mark (default: 20).
    alpha : float, optional
        Per-channel two-sided significance level (default: 0.05).
    dpi : int, optional
        Raster resolution (default: 300).
    bare : bool, optional
        Omit the caption/suptitle for embedding (default: False).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    vectors = _strand_vectors(result, None)
    fig = _draw_bihistogram_figure(
        vectors,
        title=title,
        percentage=percentage,
        width=width,
        panel_height=panel_height,
        min_sites=min_sites,
        alpha=alpha,
        bare=bare,
    )
    _save(fig, outfile, dpi)
    return fig
