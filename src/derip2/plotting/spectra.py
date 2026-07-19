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
from matplotlib.lines import Line2D
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
    DOWNSTREAM_SUBSTITUTIONS,
    SBS96_SUBSTITUTIONS,
    SBS192_SUBSTITUTIONS,
)

logger = logging.getLogger(__name__)

# Short captions distinguishing the sequence-context model in figure headings.
_CONTEXT_CAPTION = {
    'trinucleotide': r'trinucleotide context (5$^\prime$-N[R>A]N-3$^\prime$)',
    'trinucleotide_192': (
        r'trinucleotide context, strand-resolved (5$^\prime$-N[R>A]N-3$^\prime$)'
    ),
    'downstream': (
        r'downstream-triplet context '
        r'(5$^\prime$-[R>A]$d_1 d_2$-3$^\prime$; CHG methylation)'
    ),
}

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

# Two-colour encoding for the strand-asymmetry figure, so the legend swatches
# match the bars exactly (the substitution class is given by the x-axis label).
STRAND_COLORS = {
    'coding': '#2a78d6',  # blue
    'template': '#eb6834',  # orange
}

RASTER_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def _mono_bold(chars: str, bold_index: int) -> str:
    """
    Render a short motif as mathtext with one character emphasised in bold.

    matplotlib cannot bold a single character of an ordinary tick label, so the
    motif is built as a mathtext string: every base is typeset in the monospace
    math font (``\\mathtt``) except the mutated base at ``bold_index``, which is
    typeset bold (``\\mathbf``) to mark it as the substitution site.

    Parameters
    ----------
    chars : str
        The motif characters (e.g. ``'ACG'``); all must be plain ``A``/``C``/``G``/
        ``T`` so no mathtext escaping is needed.
    bold_index : int
        Index of the character to render bold.

    Returns
    -------
    str
        A mathtext string, e.g. ``r'$\\mathtt{A}\\mathbf{C}\\mathtt{G}$'``.
    """
    parts = [
        (r'\mathbf{%s}' if i == bold_index else r'\mathtt{%s}') % ch
        for i, ch in enumerate(chars)
    ]
    return '$' + ''.join(parts) + '$'


def _caption_suptitle(fig, title: Optional[str], caption: str) -> None:
    """
    Set the figure heading, distinguishing the context model via a caption line.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to title.
    title : str or None
        The caller-supplied heading, or ``None`` for the caption alone.
    caption : str
        The context-model caption (see :data:`_CONTEXT_CAPTION`).

    Returns
    -------
    None
        The suptitle is set in place.
    """
    heading = f'{title}\n{caption}' if title else caption
    fig.suptitle(heading, fontsize=TITLE_SIZE, color=INK_PRIMARY)


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


def _folded_class(ref: str, alt: str) -> str:
    """
    Return the pyrimidine-folded substitution class for a ref/alt pair.

    Parameters
    ----------
    ref : str
        Reference base.
    alt : str
        Derived base.

    Returns
    -------
    str
        One of the six pyrimidine classes, e.g. ``'C>T'``.
    """
    from derip2.spectra.channels import COMPLEMENT

    key = _substitution_label(ref, alt)
    if key in SBS6_COLORS:
        return key
    # Purine reference: fold to the pyrimidine-strand class.
    return _substitution_label(COMPLEMENT[ref], COMPLEMENT[alt])


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
    return SBS6_COLORS[_folded_class(ref, alt)]


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
    context: str = 'trinucleotide',
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
        Whether to letter every bar with its sequence-context motif.
    context : {'trinucleotide', 'downstream'}, optional
        Which context the motif ticks describe (default: ``'trinucleotide'``). For
        ``'trinucleotide'`` the motif is ``5' ref 3'`` with the middle (mutated)
        base bold; for ``'downstream'`` it is ``ref d1 d2`` with the first
        (mutated) base bold.

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
        # The mutated base is bolded in every motif: the middle base for the
        # trinucleotide context, the first base for the downstream context. The
        # two inner bases follow the canonical channel ordering (b1 outer, b2
        # inner), matching SBS96_CHANNELS / DOWNSTREAM_CHANNELS.
        downstream = context == 'downstream'
        labels = []
        for ref, _alt in substitutions:
            for b1 in BASES:
                for b2 in BASES:
                    if downstream:
                        labels.append(_mono_bold(f'{ref}{b1}{b2}', 0))
                    else:
                        labels.append(_mono_bold(f'{b1}{ref}{b2}', 1))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=4.5, color=INK_SECONDARY)
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
                # pad lifts the sample name clear of the class-block labels.
                ax.set_title(
                    result.sample_names[s],
                    fontsize=AXIS_LABEL_SIZE,
                    color=INK_PRIMARY,
                    loc='left',
                    pad=16,
                )
        _caption_suptitle(fig, title, _CONTEXT_CAPTION['trinucleotide'])
        fig.tight_layout()
        _save(fig, outfile, dpi)
    return fig


def plot_downstream(
    result,
    outfile: Optional[str] = None,
    *,
    title: Optional[str] = None,
    percentage: bool = False,
    dpi: int = 300,
):
    """
    Draw the pyrimidine-folded downstream-triplet spectrum, one panel per sample.

    The six substitution blocks mirror the SBS-96 layout, but each bar is
    classified by the mutated base plus its two downstream bases (motif
    ``ref d1 d2``, first base bold). The downstream counts are read from
    ``result.sbs96`` (which holds the 96-channel matrix for the downstream context).

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra (``result.context`` should be ``'downstream'``).
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
                DOWNSTREAM_SUBSTITUTIONS,
                percentage=percentage,
                show_context_ticks=True,
                context='downstream',
            )
            if n > 1:
                # pad lifts the sample name clear of the class-block labels.
                ax.set_title(
                    result.sample_names[s],
                    fontsize=AXIS_LABEL_SIZE,
                    color=INK_PRIMARY,
                    loc='left',
                    pad=16,
                )
        _caption_suptitle(fig, title, _CONTEXT_CAPTION['downstream'])
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
    if result.sbs192 is None:
        raise ValueError(
            'This result has no SBS-192 matrix (the downstream context has no '
            'orientation-invariant strand-resolved form); use plot_downstream.'
        )
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
                # pad lifts the sample name clear of the class-block labels.
                ax.set_title(
                    result.sample_names[s],
                    fontsize=AXIS_LABEL_SIZE,
                    color=INK_PRIMARY,
                    loc='left',
                    pad=16,
                )
        _caption_suptitle(fig, title, _CONTEXT_CAPTION['trinucleotide_192'])
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

    Raises
    ------
    ValueError
        If the result has no SBS-192 matrix (the downstream context).
    """
    if result.sbs192 is None:
        raise ValueError(
            'Strand asymmetry needs an SBS-192 matrix, which the downstream '
            'context does not produce.'
        )
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
    min_count: int = 10,
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
    min_count : int, optional
        A class is only tested (and starred) when **both** strands carry at least
        this many events (default: 10). A strand with near-zero counts gives an
        unstable binomial test, so such classes are drawn but not flagged.
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
        fig, ax = plt.subplots(figsize=(5.6, 3.2))
        fig.patch.set_facecolor(SURFACE)
        _style_axes(ax)
        x = np.arange(len(rows))
        coding = [r['coding'] for r in rows]
        template = [r['template'] for r in rows]
        # Encode strand by colour (not substitution class) so the legend swatches
        # match every bar exactly. The class is already given by the x-axis label.
        ax.bar(
            x - 0.2,
            coding,
            width=0.38,
            color=STRAND_COLORS['coding'],
            label='Coding strand',
        )
        ax.bar(
            x + 0.2,
            template,
            width=0.38,
            color=STRAND_COLORS['template'],
            label='Template strand',
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [r['class'] for r in rows], fontsize=ANNOTATION_SIZE, color=INK_SECONDARY
        )
        ax.set_ylabel('Substitutions', fontsize=AXIS_LABEL_SIZE, color=INK_PRIMARY)
        # Star classes whose coding/template split departs from 50:50 (binomial),
        # but only when both strands carry enough events for a stable test. The
        # asterisk is offset a few points above the taller bar so it hugs the bar
        # instead of floating over empty space when the class is small.
        any_star = False
        for xi, r in zip(x, rows):
            testable = min(r['coding'], r['template']) >= min_count
            if testable and r['pvalue'] < 0.05:
                any_star = True
                # Sit the asterisk's baseline on the taller bar's top so it hugs
                # the bar (near the axis for small classes) rather than floating.
                ax.text(
                    xi,
                    max(r['coding'], r['template']),
                    '*',
                    ha='center',
                    va='bottom',
                    fontsize=TITLE_SIZE,
                    color=INK_PRIMARY,
                )
        ax.legend(fontsize=LEGEND_SIZE, frameon=False, loc='upper right')
        # Explain the asterisk so the figure is self-describing.
        if any_star:
            ax.text(
                0.0,
                -0.22,
                f'* binomial p < 0.05 vs 50:50 (only classes with '
                f'>= {min_count} events on both strands are tested)',
                transform=ax.transAxes,
                fontsize=ANNOTATION_SIZE,
                color=INK_SECONDARY,
            )
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
    # The recurrence measure differs by method; name it honestly on the y-axis.
    hit_unit = (
        'independent branches'
        if result.method == 'phylogenetic'
        else 'sequences (multi-hit proxy)'
    )
    with plt.rc_context({'font.family': 'sans-serif', 'font.sans-serif': FONT_STACK}):
        fig, ax = plt.subplots(figsize=(7.4, 2.8))
        fig.patch.set_facecolor(SURFACE)
        _style_axes(ax)
        if table:
            cols = [r['col'] for r in table]
            heights = [r['n_independent'] for r in table]
            classes = [_folded_class(r['ref'], r['alt']) for r in table]
            colors = [SBS6_COLORS[c] for c in classes]
            # Translucent stems and markers so that sites stacked at the same
            # column/height stay individually visible instead of one hiding another.
            ax.vlines(cols, 0, heights, color=colors, linewidth=1.0, alpha=0.35)
            ax.scatter(
                cols,
                heights,
                s=18,
                color=colors,
                alpha=0.55,
                edgecolors='none',
                zorder=3,
            )
            ax.set_xlim(-1, result.homoplasy_counts.shape[0])
            ax.set_ylim(0, max(heights) + 1)
            # Legend maps colour to the pyrimidine-folded substitution class,
            # showing only the classes actually present.
            present = [c for c in SBS6_COLORS if c in set(classes)]
            handles = [
                Line2D(
                    [],
                    [],
                    marker='o',
                    linestyle='none',
                    markersize=5,
                    markerfacecolor=SBS6_COLORS[c],
                    markeredgecolor='none',
                    label=c,
                )
                for c in present
            ]
            # Place the legend below the axes as a single horizontal row so it
            # never overlaps the densely-packed data points.
            ax.legend(
                handles=handles,
                fontsize=LEGEND_SIZE,
                frameon=False,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.32),
                ncol=len(present),
                title='Substitution class',
                title_fontsize=LEGEND_SIZE,
                columnspacing=1.2,
                handletextpad=0.4,
            )
        else:
            ax.text(
                0.5,
                0.5,
                f'No sites hit in >= {min_hits} lineages',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=AXIS_LABEL_SIZE,
                color=INK_SECONDARY,
            )
        ax.set_xlabel('Alignment column', fontsize=AXIS_LABEL_SIZE, color=INK_PRIMARY)
        ax.set_ylabel(
            f'Independent hits\n({hit_unit})',
            fontsize=AXIS_LABEL_SIZE,
            color=INK_PRIMARY,
        )
        if title:
            ax.set_title(title, fontsize=TITLE_SIZE, color=INK_PRIMARY)
        fig.tight_layout()
        _save(fig, outfile, dpi)
    return fig
