"""
Colour theme and asset loading for the HTML reports.

The page chrome (backgrounds, text, rules) is expressed as CSS custom properties
generated from :data:`CHROME`, so every colour lives in exactly one place. The
figure surface is deliberately *not* a chrome token: inline SVG figures always
sit on the same light surface the matplotlib palette was validated against
(:data:`derip2.plotting.strandbias.SURFACE`), whatever the page looks like.
"""

from functools import cache
from importlib.resources import files

from derip2.plotting import strandbias as _palette

#: Light surface every embedded figure is drawn on. Shared with matplotlib so
#: the colourblind-safe palette holds inside the report.
FIGURE_SURFACE = _palette.SURFACE

#: Page-chrome colour tokens, emitted as ``--name`` custom properties on
#: ``:root``. Values reuse the figure ink/rule constants where the two agree.
CHROME = {
    'page': '#f9f9f7',
    'surface': _palette.SURFACE,
    'ink': _palette.INK_PRIMARY,
    'ink-2': _palette.INK_SECONDARY,
    'muted': _palette.INK_MUTED,
    'rule': _palette.GRIDLINE,
    'fig-surface': FIGURE_SURFACE,
}


@cache
def load_asset(name):
    """
    Read one packaged front-end asset as text.

    Parameters
    ----------
    name : str
        File name inside ``derip2/reporting/assets/`` (for example
        ``'base.css'``).

    Returns
    -------
    str
        The file contents.
    """
    return files('derip2.reporting').joinpath('assets', name).read_text('utf-8')


def root_css(tokens=None):
    """
    Render the ``:root`` custom-property block for a token mapping.

    Parameters
    ----------
    tokens : dict of str to str, optional
        Token name to CSS colour. Defaults to :data:`CHROME`.

    Returns
    -------
    str
        A ``:root { ... }`` CSS rule.
    """
    tokens = CHROME if tokens is None else tokens
    body = ' '.join(f'--{key}: {value};' for key, value in tokens.items())
    return f':root {{ {body} }}\n'


def base_style():
    """
    Return the stylesheet shared by every report.

    Returns
    -------
    str
        The ``:root`` token block followed by ``assets/base.css``.
    """
    return root_css() + load_asset('base.css')


def persequence_style():
    """
    Return the extra stylesheet for the per-sequence report.

    Returns
    -------
    str
        The contents of ``assets/persequence.css``.
    """
    return load_asset('persequence.css')


def persequence_script():
    """
    Return the per-sequence report's JavaScript.

    Returns
    -------
    str
        The contents of ``assets/persequence.js``.
    """
    return load_asset('persequence.js')
