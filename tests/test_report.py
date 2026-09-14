"""Tests for the alignment-wide strand-bias HTML report and its shared theme."""

import re

import pytest

from derip2.derip import DeRIP
from derip2.report import PANELS, write_html_report
from derip2.reporting import CHROME, FIGURE_SURFACE, base_style, render_page
from derip2.reporting.theme import root_css


@pytest.fixture(scope='module')
def report_html(tmp_path_factory, mintest_path):
    """Render the alignment-wide report once for the module."""
    derip = DeRIP(mintest_path)
    derip.calculate_rip()
    out = tmp_path_factory.mktemp('report') / 'report.html'
    write_html_report(derip, str(out))
    return out.read_text(encoding='utf-8')


def test_report_is_a_single_self_contained_document(report_html):
    """One <style>, inline SVG figures, and no external assets of any kind."""
    assert report_html.startswith('<!doctype html>')
    assert report_html.count('<style>') == 1
    assert '<script' not in report_html
    assert '<img' not in report_html
    assert 'src=' not in report_html
    assert not re.search(r'href="https?://', report_html)
    assert not re.search(r'@import|url\(https?:', report_html)
    assert report_html.count('<svg') == len(PANELS)


def test_report_headings_come_from_the_plot_titles(report_html):
    """Panel headings are the strand-bias MODE_TITLES, so they cannot drift."""
    from derip2.plotting.strandbias import MODE_TITLES

    for mode, heading, _blurb in PANELS:
        assert heading == MODE_TITLES[mode]
        assert f'<h2>{heading}</h2>' in report_html


def test_theme_is_single_off_white_with_figure_surface_pinned(report_html):
    """No dark-mode rules; page and surface are off-white; figures stay on
    the validated plotting surface."""
    assert 'prefers-color-scheme' not in report_html
    css = base_style()
    for token in ('page', 'surface', 'ink', 'rule', 'red', 'blue', 'yellow'):
        assert f'--{token}: {CHROME[token]};' in css
    assert CHROME['page'].lower() != '#ffffff'
    assert CHROME['surface'].lower() != '#ffffff'
    assert CHROME['fig-surface'] == FIGURE_SURFACE
    # The figure wells reference the token rather than a literal colour.
    assert re.search(r'\.figure\s*\{[^}]*var\(--fig-surface\)', css)
    # NaN cells are styled (the .muted rule exists).
    assert re.search(r'\.muted\s*\{[^}]*color', css)


def test_root_css_renders_tokens_in_order():
    """The :root block lists every token as a custom property."""
    css = root_css({'a': '#000', 'b-c': '#fff'})
    assert css.strip() == ':root { --a: #000; --b-c: #fff; }'


def test_render_page_escapes_title_and_wraps_body():
    """The shared page shell escapes the title and places body, css, scripts."""
    html = render_page(
        'A <b>',
        'sub &amp; title',
        '<p>body</p>',
        css='x{}',
        scripts='<script>1</script>',
    )
    assert '<title>A &lt;b&gt;</title>' in html
    assert '<h1>A &lt;b&gt;</h1>' in html
    assert '<p class="sub">sub &amp; title</p>' in html
    assert '<style>x{}</style>' in html
    assert (
        html.index('<p>body</p>')
        < html.index('<footer>')
        < html.index('<script>1</script>')
    )
    assert html.endswith('</main></body></html>')
