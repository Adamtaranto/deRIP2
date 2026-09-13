"""Inline-SVG embedding of matplotlib figures for the HTML reports."""

from html import escape
import io
import re


def figure_to_svg(fig, prefix, tight=True):
    """
    Render a matplotlib figure to an inline SVG fragment with namespaced IDs.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to render.
    prefix : str
        Unique string prepended to every element ID and internal reference.
    tight : bool, optional
        Trim surrounding whitespace with ``bbox_inches='tight'`` (default:
        True). Pass ``False`` to keep the figure's full, fixed canvas so a
        series of figures with the same figure size and axes rectangle render at
        identical dimensions — needed when several figures must stay aligned
        across pages (their left margins would otherwise vary with tick-label
        width).

    Returns
    -------
    str
        The ``<svg>`` element, ready to embed directly in an HTML body.

    Notes
    -----
    Matplotlib reuses the same element IDs in every SVG it writes (glyph
    definitions such as ``DejaVuSans-41``, tick group names, and so on). Several
    figures in one HTML document would therefore share IDs, and a browser
    resolves ``href="#id"`` to the *first* match in the document — so later
    figures would silently borrow the first figure's glyphs. Prefixing every ID
    and every internal reference keeps each figure self-referential.
    """
    buffer = io.StringIO()
    if tight:
        fig.savefig(buffer, format='svg', bbox_inches='tight')
    else:
        fig.savefig(buffer, format='svg')
    svg = buffer.getvalue()

    # Drop everything before the opening <svg> tag: an XML declaration or a
    # DOCTYPE inside an HTML body is invalid.
    match = re.search(r'<svg', svg)
    if match:
        svg = svg[match.start() :]

    svg = re.sub(r'\bid="([^"]+)"', rf'id="{prefix}\1"', svg)
    # Covers both href="#x" and xlink:href="#x".
    svg = re.sub(r'href="#([^"]+)"', rf'href="#{prefix}\1"', svg)
    # clip-path="url(#x)", filter="url(#x)", and friends.
    svg = re.sub(r'url\(#([^)]+)\)', rf'url(#{prefix}\1)', svg)
    return svg


def inject_svg_tooltips(svg, prefix, titles, fasta_keys=None):
    """
    Tag named ``<g>`` groups with ``data-tip`` (and optional ``data-fasta``).

    Matplotlib writes an artist's ``gid`` as ``<g id="gid">``;
    :func:`figure_to_svg` then namespaces every id with ``prefix``. Adding a
    ``data-tip`` attribute to that group's opening tag lets the report's own
    JavaScript show a floating tooltip with no delay and pin it on click (a
    native ``<title>`` would impose a ~1 s browser delay and never appear on
    click). ``aria-label`` mirrors the text so assistive technology can still
    announce it. When ``fasta_keys`` maps a ``gid`` to a payload key, a
    ``data-fasta`` attribute is added too so a click on that group opens the
    FASTA popup.

    The whole fragment is rewritten in one regular-expression pass with a
    dictionary lookup per ``<g id=...>`` tag, so the cost is linear in the SVG
    size however many groups are tagged.

    Parameters
    ----------
    svg : str
        The inline SVG fragment (already id-prefixed).
    prefix : str
        The id prefix applied by :func:`figure_to_svg`.
    titles : dict of str to str
        Maps each artist ``gid`` to its tooltip text.
    fasta_keys : dict of str to str, optional
        Maps a ``gid`` to a FASTA-payload key; those groups gain a ``data-fasta``
        attribute. ``gid``\\s present here but absent from ``titles`` are still
        tagged (with ``data-fasta`` only).

    Returns
    -------
    str
        The SVG with ``data-tip``/``aria-label``/``data-fasta`` attributes added.
    """
    fasta_keys = fasta_keys or {}
    if not titles and not fasta_keys:
        return svg

    attrs_by_gid = {}
    for gid in dict.fromkeys((*titles, *fasta_keys)):
        attrs = ''
        if gid in titles:
            esc = escape(titles[gid], quote=True)
            attrs += f' data-tip="{esc}" aria-label="{esc}"'
        if gid in fasta_keys:
            attrs += f' data-fasta="{escape(fasta_keys[gid], quote=True)}"'
        attrs_by_gid[gid] = attrs

    pattern = re.compile(rf'<g id="{re.escape(prefix)}([^"]+)">')

    def _tag(match):
        """
        Rewrite one ``<g id=...>`` opening tag, adding its attributes if tagged.

        Parameters
        ----------
        match : re.Match
            Match whose group 1 is the un-prefixed ``gid``.

        Returns
        -------
        str
            The (possibly extended) opening tag.
        """
        attrs = attrs_by_gid.get(match.group(1))
        if attrs is None:
            return match.group(0)
        return f'<g id="{prefix}{match.group(1)}"{attrs}>'

    return pattern.sub(_tag, svg)
