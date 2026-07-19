"""
Tests for the mutation-spectrum comparison statistics.

The chi-squared machinery is dependency-free (regularised incomplete gamma), so
these pin it against known values and simple constructed cases rather than SciPy.
"""

import logging
import math

import numpy as np
import pytest

from derip2.stats.spectra_compare import (
    chi2_homogeneity,
    chi2_sf,
    compare_matrix_files,
    compare_spectra,
    cosine_similarity,
    pairwise_compare,
)

logging.disable(logging.CRITICAL)


def test_cosine_similarity_bounds():
    """Identical vectors give 1, orthogonal give 0, scaling is ignored."""
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert math.isnan(cosine_similarity([0, 0], [1, 1]))


def test_cosine_length_mismatch_raises():
    """Vectors of different length are rejected."""
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_chi2_sf_known_values():
    """The chi-squared survival function matches textbook critical values."""
    # 3.841 is the 0.05 critical value for 1 dof.
    assert chi2_sf(3.841, 1) == pytest.approx(0.05, abs=1e-3)
    # 11.070 is the 0.05 critical value for 5 dof.
    assert chi2_sf(11.070, 5) == pytest.approx(0.05, abs=1e-3)
    assert chi2_sf(0.0, 4) == 1.0


def test_identical_columns_not_significant():
    """Two identical columns give chi2 = 0 and p = 1."""
    col = np.array([10, 20, 30, 40], dtype=float)
    matrix = np.column_stack([col, col])
    res = chi2_homogeneity(matrix, ['a', 'b'])
    assert res['chi2'] == pytest.approx(0.0)
    assert res['pvalue'] == pytest.approx(1.0)
    assert res['cramers_v'] == pytest.approx(0.0)


def test_proportional_columns_not_significant():
    """Columns with the same shape but different totals are homogeneous."""
    a = np.array([10, 20, 30], dtype=float)
    b = a * 5  # same proportions, 5x the depth
    res = chi2_homogeneity(np.column_stack([a, b]), ['a', 'b'])
    assert res['chi2'] == pytest.approx(0.0, abs=1e-9)
    assert res['pvalue'] == pytest.approx(1.0)


def test_divergent_columns_significant():
    """Columns with opposite shapes are flagged as different."""
    a = np.array([100, 100, 1, 1], dtype=float)
    b = np.array([1, 1, 100, 100], dtype=float)
    res = chi2_homogeneity(np.column_stack([a, b]), ['a', 'b'])
    assert res['pvalue'] < 1e-6
    assert res['cramers_v'] > 0.5


def test_empty_channels_reduce_dof():
    """Channels empty across all samples are dropped from the test."""
    # 4 channels but two are all-zero -> only 2 active channels, dof = 1.
    a = np.array([10, 0, 20, 0], dtype=float)
    b = np.array([20, 0, 10, 0], dtype=float)
    res = chi2_homogeneity(np.column_stack([a, b]), ['a', 'b'])
    assert res['n_channels_tested'] == 2
    assert res['dof'] == 1


def test_compare_spectra_reports_top_channels():
    """compare_spectra surfaces the channels driving the difference."""
    channels = ['A[C>T]A', 'A[C>T]C', 'A[C>T]G', 'A[C>T]T']
    a = np.array([100, 10, 10, 10], dtype=float)
    b = np.array([10, 10, 10, 100], dtype=float)
    res = compare_spectra(a, b, channels)
    assert res['pvalue'] < 1e-6
    assert res['cosine_similarity'] < 1.0
    # The most divergent channels are the first and last.
    top_names = {c['channel'] for c in res['top_channels'][:2]}
    assert top_names == {'A[C>T]A', 'A[C>T]T'}


def test_pairwise_compare_bonferroni():
    """Pairwise comparison returns adjusted p-values, sorted most-different first."""
    same = np.array([10, 20, 30], dtype=float)
    diff = np.array([30, 20, 10], dtype=float)
    matrix = np.column_stack([same, same, diff])
    res = pairwise_compare(matrix, ['g1', 'g2', 'g3'])
    assert len(res) == 3  # 3 choose 2
    # The g1-g2 pair is identical; its p should be ~1.
    identical = [r for r in res if {r['a'], r['b']} == {'g1', 'g2'}][0]
    assert identical['pvalue'] == pytest.approx(1.0)
    # Adjusted p-values never exceed 1 and are >= raw.
    for r in res:
        if not math.isnan(r['pvalue_adjusted']):
            assert r['pvalue_adjusted'] >= r['pvalue'] - 1e-12
            assert r['pvalue_adjusted'] <= 1.0


# ---------------------------------------------------------------------------
# File-level comparison with a context guard.
# ---------------------------------------------------------------------------


def _classes(seqs):
    """Wrap sequence strings as an object exposing ``arr`` like a classification."""
    from types import SimpleNamespace

    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    from derip2.aln_ops import alignment_to_array

    aln = MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=f'seq{i}') for i, s in enumerate(seqs)]
    )
    return SimpleNamespace(arr=alignment_to_array(aln))


def _write(result, tmp_path, name, kind):
    """Write a spectra result to a matrix file and return its path."""
    from derip2.spectra.matrix_io import write_sbs_matrix

    path = str(tmp_path / name)
    write_sbs_matrix(result, path, kind=kind)
    return path


def test_compare_matrix_files_same_context(tmp_path):
    """Two downstream matrices with matching channels compare successfully."""
    from derip2.stats import compute_spectra

    a = compute_spectra(_classes(['TCAGT', 'TTAGT']), 'TCAGT', context='downstream')
    b = compute_spectra(_classes(['TCAGT', 'TTAGT']), 'TCAGT', context='downstream')
    path_a = _write(a, tmp_path, 'a.DSC96.txt', 'downstream')
    path_b = _write(b, tmp_path, 'b.DSC96.txt', 'downstream')
    res = compare_matrix_files(path_a, path_b)
    assert res['cosine_similarity'] == pytest.approx(1.0)


def test_compare_matrix_files_cross_context_raises(tmp_path):
    """Comparing a trinucleotide matrix with a downstream one is rejected."""
    from derip2.stats import compute_spectra

    tri = compute_spectra(_classes(['ACGTA', 'ATGTA']), 'ACGTA')
    ds = compute_spectra(_classes(['TCAGT', 'TTAGT']), 'TCAGT', context='downstream')
    path_tri = _write(tri, tmp_path, 'tri.SBS96.txt', '96')
    path_ds = _write(ds, tmp_path, 'ds.DSC96.txt', 'downstream')
    with pytest.raises(ValueError):
        compare_matrix_files(path_tri, path_ds)
