"""
Smoke tests for the native SBS spectrum figures.

These confirm each plotter returns a matplotlib figure and writes a non-empty
file, and that the strand-asymmetry summary and its binomial screening p-value
behave sensibly. They do not assert on pixels.
"""

import logging
import os

import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from derip2.derip import DeRIP
from derip2.plotting import spectra as sp

logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
MINTEST = os.path.join(HERE, 'data', 'mintest.fa')


@pytest.fixture(scope='module')
def result():
    """A computed SpectraResult over mintest, shared across the plot tests."""
    d = DeRIP(MINTEST)
    d.calculate_rip()
    return d.calculate_spectra()


@pytest.fixture(scope='module')
def result_multi():
    """A per-row (multi-sample) SpectraResult over mintest."""
    d = DeRIP(MINTEST)
    d.calculate_rip()
    return d.calculate_spectra(partition_by='row')


@pytest.mark.parametrize(
    'plotter,kwargs',
    [
        (sp.plot_sbs96, {}),
        (sp.plot_sbs96, {'percentage': True}),
        (sp.plot_sbs192, {}),
        (sp.plot_strand_asymmetry, {}),
        (sp.plot_homoplasy, {}),
    ],
)
def test_plot_writes_file(tmp_path, result, plotter, kwargs):
    """Each plotter returns a figure and writes a non-empty PNG."""
    out = tmp_path / 'fig.png'
    fig = plotter(result, str(out), **kwargs)
    assert fig is not None
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)


def test_multi_sample_sbs96_has_one_panel_per_sample(result_multi):
    """A multi-sample result draws one axes per sample."""
    fig = sp.plot_sbs96(result_multi)
    assert len(fig.axes) == len(result_multi.sample_names)
    plt.close(fig)


def test_strand_asymmetry_reports_six_classes(result):
    """The strand-asymmetry summary has one row per pyrimidine class."""
    rows = sp.strand_asymmetry(result)
    assert [r['class'] for r in rows] == ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']


def test_binomial_pvalue_extremes():
    """The binomial screening p-value is small for a total split, one for even."""
    assert sp._binom_two_sided(10, 10) < 0.01
    assert sp._binom_two_sided(5, 10) == pytest.approx(1.0)
    assert sp._binom_two_sided(0, 0) == 1.0


def test_homoplasy_plot_handles_no_sites(result):
    """A homoplasy plot with an unreachable threshold still renders."""
    fig = sp.plot_homoplasy(result, min_hits=999)
    assert fig is not None
    plt.close(fig)
