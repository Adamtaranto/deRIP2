"""
Native matplotlib figures for SBS-96 / SBS-192 mutation spectra.

These reproduce the familiar SigProfiler spectrum layouts — the six-block SBS-96
bar plot, the twelve-block strand-resolved SBS-192 plot, a strand-asymmetry panel
and a homoplasy (recurrence) plot — without depending on SigProfilerPlotting. The
palette, typography and light publication surface are shared with the strand-bias
figures (:mod:`derip2.plotting.strandbias`) so the whole package renders as one
visual system.

All public functions accept a :class:`derip2.stats.mutation_spectra.SpectraResult`
and, optionally, an output path. They return the matplotlib figure so callers can
compose or further style it.
"""

import logging
import math
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from derip2.plotting.strandbias import (
    ANNOTATION_SIZE,
    AXIS_INK,
    AXIS_LABEL_SIZE,
    FONT_STACK,
    INK_PRIMARY,
    INK_SECONDARY,
    LEGEND_SIZE,
    SURFACE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
)
from derip2.spectra.channels import (
    BASES,
    SBS96_SUBSTITUTIONS,
    SBS192_SUBSTITUTIONS,
)

logger = logging.getLogger(__name__)

# The six substitution classes, coloured from the validated deRIP2 palette so the
# spectrum figures share the package's colourblind-safe hues rather than the
# unvalidated COSMIC defaults.
SBS6_COLORS = {
    'C>A': '#2a78d6',  # blue
    'C>G': '#1a1a1a',  # ink
    'C>T': '#e34948',  # red
    'T>A': '#a9a79d',  # grey
    'T>C': '#008300',  # green
    'T>G': '#eb6834',  # orange
}

RASTER_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def _substitution_label(ref: str, alt: str) -> str:
    """
    Format a substitution type as ``REF>ALT``.

    Parameters
    ----------
    ref : str
        Reference base.
    alt : str
        Derived base.

    Returns
    -------
    str
        The ``REF>ALT`` label, e.g. ``C>T``.
    """
    return f'{ref}>{alt}'


def _class_color(ref: str, alt: str) -> str:
    """
    Return the block colour for a substitution class, folding purines.

    Parameters
    ----------
    ref : str
        Reference base.
    alt : str
        Derived base.

    Returns
    -------
    str
        A hex colour from :data:`SBS6_COLORS`.
    """
    from derip2.spectra.channels import COMPLEMENT

    key = _substitution_label(ref, alt)
    if key in SBS6_COLORS:
        return SBS6_COLORS[key]
    # Purine reference: colour by the pyrimidine-folded class.
    return SBS6_COLORS[_substitution_label(COMPLEMENT[ref], COMPLEMENT[alt])]


def _save(fig, outfile: Optional[str], dpi: int) -> None:
    """
    Save a figure to ``outfile`` on the shared light surface, if a path is given.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    outfile : str or None
        Destination path; when ``None`` the figure is left open for the caller.
    dpi : int
        Raster resolution; ignored for vector formats.

    Returns
    -------
    None
        The figure is written as a side effect.
    """
    if outfile is None:
        return
    fig.savefig(outfile, dpi=dpi, bbox_inches='tight', facecolor=SURFACE)
    logger.info('Spectrum figure saved to %s', outfile)


def _style_axes(ax) -> None:
    """
    Apply the shared journal axis styling to an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style.

    Returns
    -------
    None
        The axes are styled in place.
    """
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS_INK)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS_INK, labelsize=TICK_LABEL_SIZE, length=3, width=0.8)


def _counts_for_sample(matrix: np.ndarray, sample: int, percentage: bool) -> np.ndarray:
    """
    Extract one sample's channel counts, optionally as percentages.

    Parameters
    ----------
    matrix : numpy.ndarray
        ``(n_channels, n_samples)`` count matrix.
    sample : int
        Column index of the sample to extract.
    percentage : bool
        If True, rescale so the column sums to 100.

    Returns
    -------
    numpy.ndarray
        ``(n_channels,)`` counts or percentages.
    """
    counts = matrix[:, sample].astype(float)
    if percentage:
        total = counts.sum()
        if total > 0:
            counts = 100.0 * counts / total
    return counts


def _draw_spectrum_panel(
    ax,
    counts: np.ndarray,
    substitutions: Tuple[Tuple[str, str], ...],
    *,
    percentage: bool,
    show_context_ticks: bool,
) -> None:
    """
    Draw one spectrum panel: contiguous 16-wide blocks, one per substitution type.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw into.
    counts : numpy.ndarray
        Channel counts in canonical order, length ``16 * len(substitutions)``.
    substitutions : tuple of tuple of str
        Ordered ``(ref, alt)`` substitution types, one per 16-channel block.
    percentage : bool
        Whether the y-axis is a percentage.
    show_context_ticks : bool
        Whether to letter every bar with its trinucleotide context.

    Returns
    -------
    None
        The panel is drawn in place.
    """
    n_blocks = len(substitutions)
    colors = [_class_color(ref, alt) for ref, alt in substitutions]
    x = np.arange(counts.size)
    bar_colors = np.repeat(colors, 16)
    ax.bar(x, counts, width=0.72, color=bar_colors, linewidth=0)

    _style_axes(ax)
    ax.set_xlim(-0.7, counts.size - 0.3)
    ax.set_ylabel(
        '% of substitutions' if percentage else 'Substitutions',
        fontsize=AXIS_LABEL_SIZE,
        color=INK_PRIMARY,
    )
    ax.margins(x=0)

    # Class label bars above the plot.
    ymax = ax.get_ylim()[1]
    for block, (ref, alt) in enumerate(substitutions):
        left = block * 16 - 0.5
        ax.axvspan(
            left, left + 16, ymin=0.98, ymax=1.0, color=colors[block], clip_on=False
        )
        ax.text(
            block * 16 + 7.5,
            ymax * 1.03,
            _substitution_label(ref, alt),
            ha='center',
            va='bottom',
            fontsize=ANNOTATION_SIZE,
            color=colors[block],
            fontweight='bold',
            clip_on=False,
        )

    if show_context_ticks:
        labels = []
        for ref, _alt in substitutions:
            for five in BASES:
                for three in BASES:
                    labels.append(f'{five}{ref}{three}')
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, rotation=90, fontsize=4.5, family='monospace', color=INK_SECONDARY
        )
    else:
        ax.set_xticks([block * 16 + 7.5 for block in range(n_blocks)])
        ax.set_xticklabels(
            [_substitution_label(r, a) for r, a in substitutions],
            fontsize=ANNOTATION_SIZE,
            color=INK_SECONDARY,
        )


def _figure(n_panels: int, width: float, panel_height: float):
    """
    Create a stacked-panel figure on the shared surface within a font context.

    Parameters
    ----------
    n_panels : int
        Number of vertically stacked panels (one per sample).
    width : float
        Figure width in inches.
    panel_height : float
        Height per panel in inches.

    Returns
    -------
    tuple
        ``(fig, axes)`` with ``axes`` always a 1-D array of length ``n_panels``.
    """
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(width, panel_height * n_panels), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    return fig, axes[:, 0]


def plot_sbs96(
    result,
    outfile: Optional[str] = None,
    *,
    title: Optional[str] = None,
    percentage: bool = False,
    dpi: int = 300,
):
    """
    Draw the canonical six-block SBS-96 spectrum, one panel per sample.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra.
    outfile : str or None, optional
        Output path; when ``None`` the figure is returned unsaved.
    title : str or None, optional
        Figure title.
    percentage : bool, optional
        Plot each sample as a percentage of its total (default: counts).
    dpi : int, optional
        Raster resolution (default: 300).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        n = len(result.sample_names)
        fig, axes = _figure(n, width=7.4, panel_height=2.2)
        for s, ax in enumerate(axes):
            counts = _counts_for_sample(result.sbs96, s, percentage)
            _draw_spectrum_panel(
                ax,
                counts,
                SBS96_SUBSTITUTIONS,
                percentage=percentage,
                show_context_ticks=True,
            )
            if n > 1:
                ax.set_title(
                    result.sample_names[s],
                    fontsize=AXIS_LABEL_SIZE,
                    color=INK_PRIMARY,
                    loc='left',
                )
        if title:
            fig.suptitle(title, fontsize=TITLE_SIZE, color=INK_PRIMARY)
        fig.tight_layout()
        _save(fig, outfile, dpi)
    return fig


def plot_sbs192(
    result,
    outfile: Optional[str] = None,
    *,
    title: Optional[str] = None,
    percentage: bool = False,
    dpi: int = 300,
):
    """
    Draw the strand-resolved twelve-block SBS-192 spectrum, one panel per sample.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra.
    outfile : str or None, optional
        Output path; when ``None`` the figure is returned unsaved.
    title : str or None, optional
        Figure title.
    percentage : bool, optional
        Plot each sample as a percentage of its total (default: counts).
    dpi : int, optional
        Raster resolution (default: 300).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        n = len(result.sample_names)
        fig, axes = _figure(n, width=11.0, panel_height=2.2)
        for s, ax in enumerate(axes):
            counts = _counts_for_sample(result.sbs192, s, percentage)
            _draw_spectrum_panel(
                ax,
                counts,
                SBS192_SUBSTITUTIONS,
                percentage=percentage,
                show_context_ticks=False,
            )
            if n > 1:
                ax.set_title(
                    result.sample_names[s],
                    fontsize=AXIS_LABEL_SIZE,
                    color=INK_PRIMARY,
                    loc='left',
                )
        if title:
            fig.suptitle(title, fontsize=TITLE_SIZE, color=INK_PRIMARY)
        fig.tight_layout()
        _save(fig, outfile, dpi)
    return fig


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """
    Two-sided exact binomial p-value, with a normal-approximation fallback.

    Parameters
    ----------
    k : int
        Number of successes (here, coding-strand events).
    n : int
        Number of trials (coding + template events).
    p : float, optional
        Null success probability (default: 0.5, no strand bias).

    Returns
    -------
    float
        The two-sided p-value; ``1.0`` when ``n == 0``.
    """
    if n == 0:
        return 1.0
    if n <= 1000:
        pmf_k = math.comb(n, k) * p**k * (1 - p) ** (n - k)
        tol = pmf_k * (1 + 1e-7)
        total = 0.0
        for i in range(n + 1):
            pmf_i = math.comb(n, i) * p**i * (1 - p) ** (n - i)
            if pmf_i <= tol:
                total += pmf_i
        return min(1.0, total)
    # Large n: normal approximation with continuity correction.
    mean = n * p
    sd = math.sqrt(n * p * (1 - p))
    z = (abs(k - mean) - 0.5) / sd if sd > 0 else 0.0
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def strand_asymmetry(result, sample: int = 0) -> List[dict]:
    """
    Summarise coding- versus template-strand counts per pyrimidine class.

    For each of the six pyrimidine substitution classes, the coding-strand count
    is the sum of its SBS-192 channels and the template-strand count is the sum of
    its reverse-complement purine partner's channels. A binomial test against an
    even 50/50 split gives a screening p-value for strand bias.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra.
    sample : int, optional
        Sample column to summarise (default: 0).

    Returns
    -------
    list of dict
        One dict per pyrimidine class with keys ``class``, ``coding``,
        ``template``, ``ratio`` (coding / template, ``inf`` if template is zero)
        and ``pvalue``.
    """
    counts = result.sbs192[:, sample]
    rows: List[dict] = []
    for i, (ref, alt) in enumerate(SBS96_SUBSTITUTIONS):
        coding = float(counts[i * 16 : (i + 1) * 16].sum())
        partner = i + 6  # its reverse-complement purine class, by construction
        template = float(counts[partner * 16 : (partner + 1) * 16].sum())
        total = coding + template
        ratio = coding / template if template > 0 else math.inf
        rows.append(
            {
                'class': _substitution_label(ref, alt),
                'coding': coding,
                'template': template,
                'ratio': ratio,
                'pvalue': _binom_two_sided(int(round(coding)), int(round(total))),
            }
        )
    return rows


def plot_strand_asymmetry(
    result,
    outfile: Optional[str] = None,
    *,
    sample: int = 0,
    title: Optional[str] = None,
    dpi: int = 300,
):
    """
    Plot coding- versus template-strand counts for each pyrimidine class.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra.
    outfile : str or None, optional
        Output path; when ``None`` the figure is returned unsaved.
    sample : int, optional
        Sample column to plot (default: 0).
    title : str or None, optional
        Figure title.
    dpi : int, optional
        Raster resolution (default: 300).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    rows = strand_asymmetry(result, sample)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        fig.patch.set_facecolor(SURFACE)
        _style_axes(ax)
        x = np.arange(len(rows))
        coding = [r['coding'] for r in rows]
        template = [r['template'] for r in rows]
        colors = [SBS6_COLORS[r['class']] for r in rows]
        ax.bar(x - 0.2, coding, width=0.38, color=colors, label='Coding strand')
        ax.bar(
            x + 0.2,
            template,
            width=0.38,
            color=colors,
            alpha=0.5,
            label='Template strand',
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [r['class'] for r in rows], fontsize=ANNOTATION_SIZE, color=INK_SECONDARY
        )
        ax.set_ylabel('Substitutions', fontsize=AXIS_LABEL_SIZE, color=INK_PRIMARY)
        # Star the classes with a nominally significant strand bias.
        ymax = ax.get_ylim()[1]
        for xi, r in zip(x, rows):
            if r['pvalue'] < 0.05:
                ax.text(
                    xi,
                    ymax * 0.98,
                    '*',
                    ha='center',
                    va='top',
                    fontsize=TITLE_SIZE,
                    color=INK_PRIMARY,
                )
        ax.legend(fontsize=LEGEND_SIZE, frameon=False)
        if title:
            ax.set_title(title, fontsize=TITLE_SIZE, color=INK_PRIMARY)
        fig.tight_layout()
        _save(fig, outfile, dpi)
    return fig


def plot_homoplasy(
    result,
    outfile: Optional[str] = None,
    *,
    min_hits: int = 2,
    title: Optional[str] = None,
    dpi: int = 300,
):
    """
    Plot alignment columns hit by the same substitution in >= ``min_hits`` rows.

    Each qualifying (column, derived base) is a stem whose height is the number of
    independent sequences carrying it, coloured by its pyrimidine-folded
    substitution class. This is the baseline recurrence proxy.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra.
    outfile : str or None, optional
        Output path; when ``None`` the figure is returned unsaved.
    min_hits : int, optional
        Minimum independent hits for a site to be drawn (default: 2).
    title : str or None, optional
        Figure title.
    dpi : int, optional
        Raster resolution (default: 300).

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure.
    """
    table = result.homoplasy_table(min_hits=min_hits)
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, ax = plt.subplots(figsize=(7.4, 2.6))
        fig.patch.set_facecolor(SURFACE)
        _style_axes(ax)
        if table:
            cols = [r['col'] for r in table]
            heights = [r['n_independent'] for r in table]
            colors = [_class_color(r['ref'], r['alt']) for r in table]
            ax.vlines(cols, 0, heights, color=colors, linewidth=1.4)
            ax.scatter(cols, heights, s=14, color=colors, zorder=3)
            ax.set_xlim(-1, result.homoplasy_counts.shape[0])
            ax.set_ylim(0, max(heights) + 1)
        else:
            ax.text(
                0.5,
                0.5,
                f'No sites hit in >= {min_hits} sequences',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=AXIS_LABEL_SIZE,
                color=INK_SECONDARY,
            )
        ax.set_xlabel('Alignment column', fontsize=AXIS_LABEL_SIZE, color=INK_PRIMARY)
        ax.set_ylabel('Independent hits', fontsize=AXIS_LABEL_SIZE, color=INK_PRIMARY)
        if title:
            ax.set_title(title, fontsize=TITLE_SIZE, color=INK_PRIMARY)
        fig.tight_layout()
        _save(fig, outfile, dpi)
    return fig
