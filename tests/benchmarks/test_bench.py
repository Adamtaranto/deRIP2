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

import pytest

import derip2.aln_ops as ao
from derip2.derip import DeRIP

# import tempfile


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


def test_bench_classify_columns(benchmark, sahana_alignment_small):
    """
    Benchmark the vectorised RIP column classifier in isolation.

    This is the whole-matrix boolean pass that replaced correctRIP's per-column
    Python loop. It should be a small fraction of correctRIP's total, which is
    dominated by materialising the markup position lists.
    """
    arr = ao.alignment_to_array(sahana_alignment_small)
    next_idx, prev_idx = ao._nongap_neighbors(arr)

    def run():
        return ao.classify_columns(
            arr,
            next_idx,
            prev_idx,
            max_snp_noise=0.5,
            min_rip_like=0.1,
            reaminate=False,
            progress=False,
        )

    benchmark(run)


def test_bench_classify_columns_blocked(benchmark, sahana_alignment_small):
    """Benchmark classification under column blocking, which bounds peak memory."""
    arr = ao.alignment_to_array(sahana_alignment_small)
    next_idx, prev_idx = ao._nongap_neighbors(arr)

    def run():
        return ao.classify_columns(
            arr, next_idx, prev_idx, block_size=64, progress=False
        )

    benchmark(run)


def test_bench_build_markupdict(benchmark, sahana_alignment_small):
    """Benchmark converting the classification into markup position lists."""
    cls = ao.classify_alignment(sahana_alignment_small, progress=False)
    benchmark(lambda: ao._build_markupdict(cls))


def test_bench_compute_rsi(benchmark, sahana_alignment_small):
    """Benchmark the per-sequence RIP strandedness imbalance."""
    from derip2.stats import compute_rsi

    cls = ao.classify_alignment(sahana_alignment_small, progress=False)
    benchmark(lambda: compute_rsi(cls, ambiguous='split'))


def test_bench_compute_rsi_weighted(benchmark, sahana_alignment_small):
    """The 'weight' policy builds an extra (n_rows, n_cols) float plane."""
    from derip2.stats import compute_rsi

    cls = ao.classify_alignment(sahana_alignment_small, progress=False)
    benchmark(lambda: compute_rsi(cls, ambiguous='weight'))


def test_bench_compute_spectra(benchmark, sahana_alignment_small):
    """Benchmark the tree-free SBS-96/192 assembly against a single reference."""
    from derip2.stats.mutation_spectra import compute_spectra

    cls = ao.classify_alignment(sahana_alignment_small, progress=False)
    # Use the first sequence as the ancestral reference for the benchmark.
    ancestor = str(sahana_alignment_small[0].seq)
    benchmark(lambda: compute_spectra(cls, ancestor))


def test_bench_compute_spectra_downstream(benchmark, sahana_alignment_small):
    """Benchmark the tree-free downstream-triplet assembly against a reference."""
    from derip2.stats.mutation_spectra import compute_spectra

    cls = ao.classify_alignment(sahana_alignment_small, progress=False)
    # Use the first sequence as the ancestral reference for the benchmark.
    ancestor = str(sahana_alignment_small[0].seq)
    benchmark(lambda: compute_spectra(cls, ancestor, context='downstream'))


def test_bench_compute_spectra_from_tree(benchmark, sahana_alignment_small):
    """Benchmark per-branch substitution calling over a star topology."""
    import numpy as np

    from derip2.spectra.call_mutations import compute_spectra_from_tree
    from derip2.spectra.tree_asr import TreeReconstruction

    arr = ao.alignment_to_array(sahana_alignment_small)
    n_rows, n_cols = arr.shape
    names = [f'r{i}' for i in range(n_rows)]
    node_seq = {names[i]: arr[i] for i in range(n_rows)}
    node_prob = {name: np.ones(n_cols) for name in names}
    # A star tree: every other sequence descends directly from the first.
    edges = [(names[0], names[i]) for i in range(1, n_rows)]
    rec = TreeReconstruction(
        edges=edges,
        node_seq=node_seq,
        node_prob=node_prob,
        root_name=names[0],
        tip_names=names[1:],
        n_cols=n_cols,
    )
    benchmark(lambda: compute_spectra_from_tree(rec))


def test_bench_plot_strand_bias(benchmark, sahana_alignment_small):
    """Benchmark rendering the strand-bias figure to an in-memory SVG."""
    import io

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from derip2.plotting.strandbias import plot_strand_bias

    cls = ao.classify_alignment(sahana_alignment_small, progress=False)

    def run():
        fig = plot_strand_bias(cls, columns='rip', max_columns=10_000)
        buffer = io.BytesIO()
        fig.savefig(buffer, format='svg')
        plt.close(fig)

    benchmark(run)


def test_bench_per_sequence_report(benchmark, sahana_alignment_small, tmp_path):
    """
    Benchmark building the per-sequence HTML report over a realistic subset.

    This exercises the whole per-sequence path — the row strip, the fixed-height
    strand-bias strip and the per-sequence SBS-96 panel rendered to inline SVG
    for every sequence, plus the stats slices — which is the dominant cost of
    ``--per-seq-report`` on real data.
    """
    d = DeRIP(sahana_alignment_small)
    d.calculate_rip()
    out = str(tmp_path / 'per_seq.html')

    # Cap to keep the benchmark bounded; the per-panel work is what we measure.
    benchmark(lambda: d.write_per_sequence_report(out, max_seqs=10))


SAHANA_GFF = os.path.join(HERE, '..', 'data', 'sahana_prime_miniprot_KAH4331560.gff3')


@pytest.fixture(scope='session')
def sahana_cds_setup(sahana_alignment):
    """
    Full alignment + parsed Sahana_prime gene + its CDS alignment columns.

    Uses the full alignment because the miniprot CDS lives near columns
    2874–5381 — the trimmed [:40, :500] subset does not contain it.
    """
    from derip2.annotation import (
        cds_alignment_columns,
        parse_gff3,
        ungapped_to_column_map,
    )

    d = DeRIP(sahana_alignment)
    d.calculate_rip()
    genes = parse_gff3(SAHANA_GFF)
    ids = [r.id for r in d.alignment]
    ri = ids.index('Sahana_prime')
    u2c = ungapped_to_column_map(d.column_classes.arr[ri])
    gene = genes['Sahana_prime'][0]
    cols = cds_alignment_columns(gene, u2c)
    return d.column_classes.arr, gene, cols


def test_bench_parse_gff3(benchmark):
    """Benchmark GFF3 parsing of the multi-exon Sahana_prime gene model."""
    from derip2.annotation import parse_gff3

    benchmark(lambda: parse_gff3(SAHANA_GFF))


def test_bench_cds_extraction_across_records(benchmark, sahana_cds_setup):
    """
    Benchmark projecting a CDS onto every alignment record and calling stops.

    This is the cross-record extraction the per-sequence report's CDS annotation
    track drives (396 sequences × ~2000 CDS columns); it must stay vectorised.
    """
    from derip2.annotation import cds_stop_columns

    arr, gene, cols = sahana_cds_setup

    def run():
        return [cds_stop_columns(gene, arr[i], cols) for i in range(arr.shape[0])]

    benchmark(run)
