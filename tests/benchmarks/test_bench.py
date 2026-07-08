"""
CodSpeed performance benchmarks for the deRIP2 hot paths.

Run with::

    pytest tests/benchmarks --codspeed

These benchmarks use the large real-world alignment ``tests/data/sahana.fasta.gz``
(396 rows x 5812 columns) so that per-function timings reflect realistic load.
The ``benchmark`` fixture is provided by ``pytest-codspeed``; when the plugin is
not active (plain ``pytest``) the tests are skipped.
"""

import gzip
import logging
import os
import tempfile

import pytest

import derip2.aln_ops as ao
from derip2.derip import DeRIP

pytest_codspeed = pytest.importorskip('pytest_codspeed')

# Keep logging quiet so it does not distort timings
logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
SAHANA = os.path.join(HERE, '..', 'data', 'sahana.fasta.gz')


@pytest.fixture(scope='session')
def sahana_alignment():
    """Full realistic alignment — used only in nightly/manual deep-benchmark runs. Load the large sahana alignment once for the whole session."""
    with gzip.open(SAHANA, 'rt') as fh:
        from Bio import AlignIO

        return AlignIO.read(fh, 'fasta')


@pytest.fixture(scope='session')
def sahana_alignment_small(sahana_alignment):
    """Trimmed subset for fast per-PR CI benchmarking."""
    trimmed = sahana_alignment[:40, :500]
    return trimmed


def test_bench_calculate_rip(benchmark, sahana_alignment_small):
    """Benchmark the full RIP detection/correction pipeline (correctRIP-driven)."""

    def run():
        d = DeRIP(sahana_alignment_small)
        d.calculate_rip()
        return d

    benchmark(run)


def test_bench_fill_conserved(benchmark, sahana_alignment_small):
    """Benchmark the conserved-column pre-fill pass."""
    tracker = ao.initTracker(sahana_alignment_small)
    benchmark(lambda: ao.fillConserved(sahana_alignment_small, tracker, 0.7))


def test_bench_correct_rip(benchmark, sahana_alignment_small):
    """Benchmark correctRIP in isolation (excludes fill/colorize/plot)."""
    tracker = ao.initTracker(sahana_alignment_small)
    tracker = ao.fillConserved(sahana_alignment_small, tracker, 0.7)
    rip_counts = ao.initRIPCounter(sahana_alignment_small)

    def run():
        return ao.correctRIP(
            sahana_alignment_small,
            tracker,
            rip_counts,
            max_snp_noise=0.5,
            min_rip_like=0.1,
            reaminate=False,
            mask=True,
        )

    benchmark(run)


def test_bench_plot_alignment(benchmark, sahana_alignment_small):
    """Benchmark rendering the alignment plot with RIP markup."""
    d = DeRIP(sahana_alignment_small)
    d.calculate_rip()
    tmp = tempfile.mktemp(suffix='.png')

    def run():
        d.plot_alignment(tmp)

    try:
        benchmark(run)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
