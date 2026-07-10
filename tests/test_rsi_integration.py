"""Tests for the DeRIP strand-bias methods, the stats table and the HTML report."""

import logging
import math
import os
import re

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402

from derip2.derip import DeRIP  # noqa: E402
from derip2.report import _format_cell  # noqa: E402

logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
MINTEST = os.path.join(HERE, 'data', 'mintest.fa')


@pytest.fixture(autouse=True)
def close_figures():
    """Never leak figures between tests."""
    yield
    plt.close('all')


@pytest.fixture
def derip():
    """A DeRIP object with RIP already calculated."""
    d = DeRIP(MINTEST)
    d.calculate_rip()
    return d


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    'method,args',
    [
        ('calculate_rsi', ()),
        ('summarize_stats', ()),
        ('plot_strand_bias', ()),
        ('write_html_report', ('out.html',)),
    ],
)
def test_methods_require_calculate_rip(method, args):
    """Every strand-bias entry point fails loudly before calculate_rip."""
    d = DeRIP(MINTEST)
    with pytest.raises(ValueError, match='Must call calculate_rip'):
        getattr(d, method)(*args)


def test_calculate_rip_caches_the_classification(derip):
    """The classification is stored so stats and figures reuse it."""
    assert derip.column_classes is not None
    assert derip.column_classes.arr.shape == (
        len(derip.alignment),
        derip.alignment.get_alignment_length(),
    )


# --------------------------------------------------------------------------
# RSI on the class
# --------------------------------------------------------------------------
def test_calculate_rsi_annotates_records(derip):
    """RSI and its components land on each SeqRecord, like CRI does."""
    derip.calculate_rsi()
    for i, record in enumerate(derip.alignment):
        assert record.annotations['RSI'] == pytest.approx(derip.rsi_result.rsi[i])
        assert 'p_fwd' in record.annotations
        assert 'p_rev' in record.annotations


def test_get_rsi_values_computes_on_demand(derip):
    """Fetching values without calling calculate_rsi first still works."""
    assert derip.rsi_result is None
    values = derip.get_rsi_values()
    assert len(values) == len(derip.alignment)
    assert values[0]['id'] == derip.alignment[0].id


def test_sort_by_rsi_orders_forward_to_reverse(derip):
    """Descending sort runs from most forward-strand to most reverse-strand RIP."""
    ordered = derip.sort_by_rsi(descending=True)
    scores = [
        r.annotations['RSI'] for r in ordered if not math.isnan(r.annotations['RSI'])
    ]
    assert scores == sorted(scores, reverse=True)

    ascending = derip.sort_by_rsi(descending=False)
    scores = [
        r.annotations['RSI'] for r in ascending if not math.isnan(r.annotations['RSI'])
    ]
    assert scores == sorted(scores)


def test_sort_by_rsi_places_undefined_scores_last(derip):
    """A sequence with no evidence sorts to the end, not to an extreme."""
    derip.calculate_rsi()
    # Force one row's RSI to NaN, as happens when a strand has no sites at all.
    derip.alignment[2].annotations['RSI'] = float('nan')

    for descending in (True, False):
        ordered = derip.sort_by_rsi(descending=descending)
        # get_rsi_values() short-circuits because rsi_result is cached, so the
        # patched annotation survives.
        assert math.isnan(ordered[-1].annotations['RSI'])


def test_sort_by_rsi_inplace_invalidates_results(derip):
    """Reordering rows discards results keyed to the old order."""
    derip.sort_by_rsi(inplace=True)
    assert derip.column_classes is None
    assert derip.rsi_result is None
    assert derip.consensus is None

    with pytest.raises(ValueError, match='Must call calculate_rip'):
        derip.summarize_stats()


def test_sort_by_rsi_does_not_change_the_consensus():
    """The deRIP'd sequence is a property of the columns, not the row order."""
    plain = DeRIP(MINTEST)
    plain.calculate_rip()
    expected = str(plain.consensus.seq)

    sorted_d = DeRIP(MINTEST)
    sorted_d.calculate_rip()
    sorted_d.sort_by_rsi(inplace=True)
    sorted_d.fill_index = None
    sorted_d.calculate_rip()

    assert str(sorted_d.consensus.seq) == expected


# --------------------------------------------------------------------------
# Stats table
# --------------------------------------------------------------------------
# Golden values for tests/data/mintest.fa under the default 'split' policy.
# These pin the whole chain: column classification -> product/substrate
# attribution -> ambiguity resolution -> RSI. A change to the RIP detection
# rules that leaves the consensus untouched will still be caught here.
MINTEST_RSI = [0.241758, 0.549451, 0.311688, 0.0, -0.25, 0.0]
MINTEST_P_FWD = [0.384615, 0.692308, 0.454545, 0.0, 0.0, 0.0]
MINTEST_P_REV = [0.142857, 0.142857, 0.142857, 0.0, 0.25, 0.0]


def test_mintest_rsi_matches_golden_values(derip):
    """Regression: the full RSI chain on the reference alignment."""
    result = derip.calculate_rsi()
    assert np.allclose(result.rsi, MINTEST_RSI, atol=1e-6)
    assert np.allclose(result.p_fwd, MINTEST_P_FWD, atol=1e-6)
    assert np.allclose(result.p_rev, MINTEST_P_REV, atol=1e-6)

    pooled = result.pooled()
    assert pooled['RSI'] == pytest.approx(0.181197, abs=1e-6)
    assert pooled['pvalue'] == pytest.approx(0.109664, abs=1e-6)
    assert pooled['n_ambiguous'] == 3


def test_mintest_components_separate_unripped_from_saturated(derip):
    """RSI 0 is read together with its components, as the docs instruct."""
    result = derip.calculate_rsi()
    # Seq4 and Seq6 score RSI 0 because neither strand was touched at all,
    # not because both were RIP'd to exhaustion.
    for row in (3, 5):
        assert result.rsi[row] == pytest.approx(0.0)
        assert result.p_fwd[row] == pytest.approx(0.0)
        assert result.p_rev[row] == pytest.approx(0.0)

    # Seq2 carries the strongest forward signal; Seq5 the only reverse one.
    assert result.rsi.argmax() == 1
    assert result.rsi.argmin() == 4
    assert result.rsi[4] < 0


def test_summarize_stats_has_one_row_per_sequence(derip):
    """The table covers every sequence and every documented statistic."""
    df = derip.summarize_stats()
    assert len(df) == len(derip.alignment)
    assert list(df['ID']) == [r.id for r in derip.alignment]

    expected = {
        'index',
        'ID',
        'GC',
        'CRI',
        'PI',
        'SI',
        'RSI',
        'p_fwd',
        'p_rev',
        'fwd_product',
        'fwd_substrate',
        'rev_product',
        'rev_substrate',
        'n_ambiguous',
        'RIP_fwd',
        'RIP_rev',
        'non_RIP',
        'pvalue',
    }
    assert expected <= set(df.columns)


def test_summarize_stats_matches_rip_counts(derip):
    """The table's RIP columns agree with the alignment scan's own counters."""
    df = derip.summarize_stats()
    for i in range(len(derip.alignment)):
        assert df.loc[i, 'RIP_fwd'] == derip.rip_counts[i].RIPcount
        assert df.loc[i, 'RIP_rev'] == derip.rip_counts[i].revRIPcount
        assert df.loc[i, 'non_RIP'] == derip.rip_counts[i].nonRIPcount


def test_summarize_stats_rsi_equals_p_fwd_minus_p_rev(derip):
    """The headline statistic is exactly the difference of its components."""
    df = derip.summarize_stats()
    finite = df.dropna(subset=['RSI'])
    assert np.allclose(finite['RSI'], finite['p_fwd'] - finite['p_rev'])


def test_summarize_stats_honours_the_ambiguity_policy(derip):
    """A different policy recomputes RSI rather than reusing the cached result."""
    split = derip.summarize_stats(ambiguous='split')
    # No manual invalidation: the cached policy must be checked, not just the
    # presence of a cached result.
    excluded = derip.summarize_stats(ambiguous='exclude')

    assert derip.rsi_result.ambiguous == 'exclude'
    assert split['fwd_product'].sum() > excluded['fwd_product'].sum()

    # And switching back recomputes again.
    back = derip.summarize_stats(ambiguous='split')
    assert np.allclose(back['fwd_product'], split['fwd_product'])


def test_write_stats_honours_the_ambiguity_policy(derip, tmp_path):
    """The policy reaches the file, not just the in-memory table."""
    import pandas as pd

    derip.summarize_stats(ambiguous='split')  # prime the cache with another policy
    out = tmp_path / 'stats.tsv'
    derip.write_stats(str(out), ambiguous='both')

    loaded = pd.read_csv(out, sep='\t')
    assert derip.rsi_result.ambiguous == 'both'
    # 'both' double-counts ambiguous events, so products exceed the split total.
    assert (
        loaded['fwd_product'].sum()
        > derip.summarize_stats(ambiguous='split')['fwd_product'].sum()
    )


def test_write_stats_round_trips(derip, tmp_path):
    """The TSV holds the same numbers as the DataFrame."""
    import pandas as pd

    out = tmp_path / 'stats.tsv'
    derip.write_stats(str(out))
    loaded = pd.read_csv(out, sep='\t')

    assert list(loaded['ID']) == list(derip.summarize_stats()['ID'])
    assert np.allclose(
        loaded['RSI'].dropna(), derip.summarize_stats()['RSI'].dropna(), atol=1e-5
    )


def test_stats_summary_is_printable(derip):
    """The terminal summary is a plain string naming every sequence."""
    text = derip.stats_summary()
    assert isinstance(text, str)
    for record in derip.alignment:
        assert record.id in text
    assert 'RSI' in text


# --------------------------------------------------------------------------
# Plot method
# --------------------------------------------------------------------------
def test_plot_strand_bias_supplies_the_consensus_automatically(derip):
    """xaxis='derip' works without the caller passing the sequence."""
    fig = derip.plot_strand_bias(xaxis='derip')
    assert fig is not None


def test_plot_strand_bias_writes_a_file(derip, tmp_path):
    """The figure is written where asked."""
    out = tmp_path / 'bias.svg'
    derip.plot_strand_bias(output_file=str(out))
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------
def test_html_report_is_self_contained(derip, tmp_path):
    """No external asset is ever fetched: no src=, no remote href, no <img>."""
    out = tmp_path / 'report.html'
    derip.write_html_report(str(out))
    html = out.read_text()

    assert '<img' not in html
    assert 'src=' not in html
    # The only absolute URLs are XML namespaces and metadata, never fetched.
    for url in re.findall(r'(?:href|src)="(https?://[^"]+)"', html):
        pytest.fail(f'remote reference in report: {url}')


def test_html_report_embeds_all_three_panels(derip, tmp_path):
    """RIP, non-RIP and all-deamination figures are each present as inline SVG."""
    out = tmp_path / 'report.html'
    derip.write_html_report(str(out))
    html = out.read_text()

    assert html.count('<svg') == 3
    assert '<?xml' not in html, 'XML prolog must be stripped for HTML embedding'
    for heading in ('RIP-like mutations', 'Non-RIP deamination', 'All C/G deamination'):
        assert heading in html


def test_html_report_svg_ids_are_unique(derip, tmp_path):
    """Namespacing prevents a later figure borrowing an earlier figure's glyphs."""
    out = tmp_path / 'report.html'
    derip.write_html_report(str(out))
    html = out.read_text()

    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), 'duplicate element IDs across embedded SVGs'

    # Every internal reference resolves to an ID defined in this document.
    refs = set(re.findall(r'href="#([^"]+)"', html))
    refs |= set(re.findall(r'url\(#([^)]+)\)', html))
    assert refs <= set(ids)


def test_html_report_contains_every_sequence(derip, tmp_path):
    """The statistics table lists all sequences."""
    out = tmp_path / 'report.html'
    derip.write_html_report(str(out))
    html = out.read_text()
    for record in derip.alignment:
        assert record.id in html


def test_html_report_survives_an_undrawable_panel(derip, tmp_path):
    """A panel that cannot be drawn is noted inline, not fatal."""
    out = tmp_path / 'report.html'
    derip.write_html_report(str(out), columns='all', max_columns=2)
    html = out.read_text()
    assert 'Not drawn' in html
    # The table is still there.
    assert 'Per-sequence statistics' in html


def test_report_formats_nan_as_a_dash():
    """An undefined statistic renders as an en-dash, never as 'nan'."""
    text, css = _format_cell('RSI', float('nan'))
    assert text == '&ndash;'
    assert css == 'muted'


def test_report_signs_rsi_and_colours_by_direction():
    """RSI carries an explicit sign and a direction class."""
    assert _format_cell('RSI', 0.5) == ('+0.500', 'pos')
    assert _format_cell('RSI', -0.5) == ('-0.500', 'neg')
    assert _format_cell('RSI', 0.0) == ('+0.000', '')


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_cli_writes_strand_bias_outputs(tmp_path):
    """The new flags produce a figure, a stats table and a report."""
    from click.testing import CliRunner

    from derip2.app import main

    result = CliRunner().invoke(
        main,
        [
            '-i',
            MINTEST,
            '-d',
            str(tmp_path),
            '-p',
            'demo',
            '--plot-strand-bias',
            '--stats-out',
            '--html-report',
            '--sort-by-rsi',
            '--rsi-ambiguous',
            'weight',
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / 'demo_strand_bias.svg').exists()
    assert (tmp_path / 'demo_stats.tsv').exists()
    assert (tmp_path / 'demo_report.html').exists()
    assert (tmp_path / 'demo.fasta').exists()


def test_cli_sorting_preserves_the_consensus(tmp_path):
    """--sort-by-rsi reorders rows without changing the deRIP'd sequence."""
    from click.testing import CliRunner

    from derip2.app import main

    runner = CliRunner()
    for name, extra in (('plain', []), ('sorted', ['--sort-by-rsi'])):
        result = runner.invoke(
            main, ['-i', MINTEST, '-d', str(tmp_path), '-p', name, *extra]
        )
        assert result.exit_code == 0, result.output

    def consensus(name):
        text = (tmp_path / f'{name}.fasta').read_text()
        return ''.join(text.splitlines()[1:])

    assert consensus('plain') == consensus('sorted')
