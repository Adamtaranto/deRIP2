"""
Sequence-logo glyphs drawn with matplotlib text paths.

Letters are rendered as filled paths and scaled independently in x and y, which
is what distinguishes a sequence logo from a row of text: glyph height encodes
information content, so an 'A' worth 1.5 bits is drawn one-and-a-half times as
tall as an 'A' worth 1 bit while keeping the same width.

This avoids a dependency on a dedicated logo package; matplotlib is already
required.
"""

from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np

# Cache of unit-normalised glyph paths, keyed by character. Building a TextPath
# is comparatively slow and a logo re-draws the same five characters thousands
# of times.
_GLYPH_CACHE = {}

BASES = ('A', 'C', 'G', 'T')


def _unit_glyph(char, font=None):
    """
    Return a glyph path normalised into the unit square.

    Parameters
    ----------
    char : str
        Single character to render.
    font : matplotlib.font_manager.FontProperties, optional
        Font used to trace the glyph. Defaults to a bold sans face.

    Returns
    -------
    matplotlib.path.Path
        Path whose bounding box is exactly ``(0, 0)`` to ``(1, 1)``.
    """
    if char in _GLYPH_CACHE:
        return _GLYPH_CACHE[char]

    font = font or FontProperties(family='sans-serif', weight='bold')
    path = TextPath((0, 0), char, size=1, prop=font)

    bounds = path.get_extents()
    width = bounds.width or 1.0
    height = bounds.height or 1.0

    # Translate to the origin, then scale the bounding box to 1x1.
    normalise = (
        Affine2D().translate(-bounds.x0, -bounds.y0).scale(1.0 / width, 1.0 / height)
    )
    unit = normalise.transform_path(path)
    _GLYPH_CACHE[char] = unit
    return unit


def draw_glyph(ax, char, x, y, width, height, color, alpha=1.0, zorder=3):
    """
    Draw a single letter stretched to fill a given rectangle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    char : str
        Character to draw.
    x, y : float
        Lower-left corner of the target rectangle, in data coordinates.
    width, height : float
        Size of the target rectangle, in data coordinates.
    color : str
        Fill colour.
    alpha : float, optional
        Fill opacity (default: 1.0).
    zorder : int, optional
        Drawing order (default: 3).

    Returns
    -------
    matplotlib.patches.PathPatch or None
        The patch added to the axes, or None when the rectangle has no area.
    """
    if height <= 0 or width <= 0:
        return None

    transform = Affine2D().scale(width, height).translate(x, y)
    patch = PathPatch(
        transform.transform_path(_unit_glyph(char)),
        facecolor=color,
        edgecolor='none',
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_gap(ax, x, y, width, height, color, alpha=1.0, zorder=3):
    """
    Draw a short centred rule standing for a gap in the consensus.

    A ``'-'`` routed through :func:`draw_glyph` would be normalised to the unit
    square and stretched into a solid block, so the dash is drawn directly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    x, y : float
        Centre of the rule, in data coordinates.
    width, height : float
        Size of the rule, in data coordinates.
    color : str
        Fill colour.
    alpha : float, optional
        Fill opacity (default: 1.0).
    zorder : int, optional
        Drawing order; matches :func:`draw_glyph` so a dash and a letter share a
        layer (default: 3).

    Returns
    -------
    matplotlib.patches.PathPatch or None
        The patch added to the axes, or None when the rule has no area. A
        PathPatch rather than a Rectangle so it cannot be mistaken for one of
        the ``axvspan`` region markers, which are Rectangles.
    """
    if height <= 0 or width <= 0:
        return None

    left, right = x - width / 2.0, x + width / 2.0
    bottom, top = y - height / 2.0, y + height / 2.0
    # Path(closed=True) discards the final vertex in favour of a CLOSEPOLY, so
    # the first corner has to be repeated or the rule collapses to a triangle.
    corners = [(left, bottom), (left, top), (right, top), (right, bottom)]
    patch = PathPatch(
        Path(corners + corners[:1], closed=True),
        facecolor=color,
        edgecolor='none',
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def column_information(counts, pseudocount=0.0):
    """
    Shannon information content of an alignment column, in bits.

    Parameters
    ----------
    counts : array_like
        Counts of A, C, G, T in the column.
    pseudocount : float, optional
        Added to every base count before computing frequencies, which keeps the
        entropy finite for very shallow columns (default: 0.0).

    Returns
    -------
    tuple
        ``(information, frequencies)`` where ``information`` is
        ``log2(4) - H`` in bits (0 to 2) and ``frequencies`` sums to 1. Both are
        zero for an empty column.

    Notes
    -----
    No small-sample correction is applied. For alignments with few sequences the
    information content is biased upward; pass a ``pseudocount`` to temper this.
    """
    counts = np.asarray(counts, dtype=float) + pseudocount
    total = counts.sum()
    if total <= 0:
        return 0.0, np.zeros(4)

    freqs = counts / total
    nonzero = freqs[freqs > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero))
    return float(np.log2(4) - entropy), freqs


def draw_logo_column(
    ax, counts, x, y0, max_height, colors, bar_width=0.8, pseudocount=0.0, alpha=1.0
):
    """
    Draw one stacked sequence-logo column, tallest base on top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    counts : array_like
        Counts of A, C, G, T in the column.
    x : float
        Centre of the column in data coordinates.
    y0 : float
        Baseline of the column.
    max_height : float
        Height corresponding to the maximum 2 bits of information.
    colors : dict
        Mapping of base character to colour.
    bar_width : float, optional
        Glyph width in data coordinates (default: 0.8).
    pseudocount : float, optional
        Passed to :func:`column_information` (default: 0.0).
    alpha : float, optional
        Fill opacity of every glyph in the stack (default: 1.0).

    Returns
    -------
    float
        Total height drawn, in data coordinates.
    """
    information, freqs = column_information(counts, pseudocount)
    if information <= 0:
        return 0.0

    total_height = max_height * information / 2.0

    # Stack smallest first so the dominant base ends up on top, as is
    # conventional for sequence logos.
    order = np.argsort(freqs)
    cursor = y0
    for i in order:
        height = total_height * freqs[i]
        if height <= 0:
            continue
        base = BASES[i]
        draw_glyph(
            ax,
            base,
            x - bar_width / 2.0,
            cursor,
            bar_width,
            height,
            colors[base],
            alpha=alpha,
        )
        cursor += height

    return total_height
