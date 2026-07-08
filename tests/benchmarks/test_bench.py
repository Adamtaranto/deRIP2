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
    """Load the large sahana alignment once for the whole session."""
    with gzip.open(SAHANA, 'rt') as fh:
        from Bio import AlignIO

        return AlignIO.read(fh, 'fasta')


def test_bench_calculate_rip(benchmark, sahana_alignment):
    """Benchmark the full RIP detection/correction pipeline (correctRIP-driven)."""

    def run():
        d = DeRIP(sahana_alignment)
        d.calculate_rip()
        return d

    benchmark(run)


def test_bench_fill_conserved(benchmark, sahana_alignment):
    """Benchmark the conserved-column pre-fill pass."""
    tracker = ao.initTracker(sahana_alignment)
    benchmark(lambda: ao.fillConserved(sahana_alignment, tracker, 0.7))


def test_bench_correct_rip(benchmark, sahana_alignment):
    """Benchmark correctRIP in isolation (excludes fill/colorize/plot)."""
    tracker = ao.initTracker(sahana_alignment)
    tracker = ao.fillConserved(sahana_alignment, tracker, 0.7)
    rip_counts = ao.initRIPCounter(sahana_alignment)

    def run():
        return ao.correctRIP(
            sahana_alignment,
            tracker,
            rip_counts,
            max_snp_noise=0.5,
            min_rip_like=0.1,
            reaminate=False,
            mask=True,
        )

    benchmark(run)


def test_bench_plot_alignment(benchmark, sahana_alignment):
    """Benchmark rendering the alignment plot with RIP markup."""
    d = DeRIP(sahana_alignment)
    d.calculate_rip()
    tmp = tempfile.mktemp(suffix='.png')

    def run():
        d.plot_alignment(tmp)

    try:
        benchmark(run)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
