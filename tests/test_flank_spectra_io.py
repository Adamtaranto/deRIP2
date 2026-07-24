"""
Tests for the DeRIP flank-context spectra API and the TSV writers.

Exercises the caching methods on ``DeRIP`` and the tidy matrix / comparison
file formats on the committed ``mintest`` alignment.
"""

import csv

import pytest

from derip2.derip import DeRIP
from derip2.stats.flank_spectra import (
    COMPARISON_KEYS,
    write_flank_comparisons,
    write_flank_matrix,
)


@pytest.fixture
def mintest_derip(mintest_path):
    """A DeRIP object with RIP already calculated on the mintest alignment."""
    d = DeRIP(mintest_path)
    d.calculate_rip()
    return d


def test_calculate_flank_spectra_caches_and_labels(mintest_derip):
    """calculate_flank_spectra returns a result cached on the object."""
    result = mintest_derip.calculate_flank_spectra()
    assert result is mintest_derip.flank_spectra_result
    # One sample column per alignment sequence, labelled by sequence id.
    ids = [rec.id for rec in mintest_derip.alignment]
    assert result.sample_names == ids
    assert result.sub_fwd.shape == (16, len(ids))


def test_calculate_flank_spectra_requires_rip(mintest_path):
    """The method raises if RIP has not been calculated."""
    d = DeRIP(mintest_path)
    with pytest.raises(ValueError):
        d.calculate_flank_spectra()


def test_write_matrix_computes_on_demand(mintest_path, tmp_path):
    """write_flank_spectra_matrix computes spectra if not already done."""
    d = DeRIP(mintest_path)
    d.calculate_rip()
    out = tmp_path / 'ctx.tsv'
    d.write_flank_spectra_matrix(str(out))
    assert d.flank_spectra_result is not None
    with open(out) as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    assert rows[0].keys() >= {'sample', 'state', 'strand', 'channel', 'count'}
    # 6 (state x strand) blocks of 16 channels per sample.
    n_samples = len(d.alignment)
    assert len(rows) == n_samples * 6 * 16


def test_matrix_combined_equals_forward_plus_reverse(mintest_derip, tmp_path):
    """The 'combined' rows equal the sum of forward and reverse per channel."""
    result = mintest_derip.calculate_flank_spectra()
    out = tmp_path / 'ctx.tsv'
    write_flank_matrix(result, str(out))
    with open(out) as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))

    # Index counts by (sample, state, strand, channel).
    counts = {
        (r['sample'], r['state'], r['strand'], r['channel']): float(r['count'])
        for r in rows
    }
    for sample in result.sample_names:
        for state in ('substrate', 'product'):
            for channel in (
                result.channels_substrate
                if state == 'substrate'
                else result.channels_product
            ):
                combined = counts[(sample, state, 'combined', channel)]
                fwd = counts[(sample, state, 'forward', channel)]
                rev = counts[(sample, state, 'reverse', channel)]
                assert combined == fwd + rev


def test_write_comparisons_one_row_per_sample_and_comparison(mintest_derip, tmp_path):
    """The comparison TSV has n_samples x 5 rows with the expected columns."""
    out = tmp_path / 'cmp.tsv'
    mintest_derip.write_flank_spectra_comparisons(str(out))
    with open(out) as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    n_samples = len(mintest_derip.alignment)
    assert len(rows) == n_samples * len(COMPARISON_KEYS)
    assert set(rows[0].keys()) == {
        'sample',
        'comparison',
        'cosine',
        'cramers_v',
        'chi2',
        'dof',
        'pvalue',
        'n_a',
        'n_b',
        'chi2_reliable',
        'top_channels',
    }
    assert {r['comparison'] for r in rows} == set(COMPARISON_KEYS)


def test_write_comparisons_direct_helper(mintest_derip, tmp_path):
    """The module-level writer works without going through DeRIP."""
    result = mintest_derip.calculate_flank_spectra()
    out = tmp_path / 'cmp.tsv'
    returned = write_flank_comparisons(result, str(out), min_sites=5)
    assert returned == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_flank_spectra_method_returns_figure(mintest_derip):
    """DeRIP.plot_flank_spectra returns the pooled three-panel bihistogram."""
    import matplotlib.pyplot as plt

    fig = mintest_derip.plot_flank_spectra()
    assert len(fig.axes) == 3
    plt.close(fig)
