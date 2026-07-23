"""
Unit tests for flanking-context spectra of RIP-like sites.

The counting/folding logic is exercised on tiny, hand-checkable alignments where
the expected channel of every site can be worked out by eye. Channel 11 (=
``up=G, down=T``) recurs throughout because ``GCAT``/``GTAT`` motifs are the
easiest to trace; the reverse-strand cases are constructed so their fold lands on
that same channel, which is exactly what the swap-and-complement rule predicts.
"""

import numpy as np
import pytest
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from derip2.aln_ops import classify_alignment
from derip2.spectra.flank_channels import (
    COMP_CODE,
    FLANK16_LABELS_CA,
    FLANK16_LABELS_TA,
    FLANK16_PAIR_LABELS,
    IDX16_TABLE,
    flank_channel_labels,
)
from derip2.stats.flank_spectra import (
    COMPARISON_KEYS,
    FlankSpectraResult,
    compare_flank_spectra,
    compute_flank_spectra,
)

# Channel index for the up=G, down=T flank pair (2*4 + 3), used across tests.
GT_CHANNEL = 11


def make_alignment(seqs, ids=None):
    """Build a MultipleSeqAlignment from a list of sequence strings."""
    if ids is None:
        ids = [f'seq{i}' for i in range(len(seqs))]
    return MultipleSeqAlignment([SeqRecord(Seq(s), id=i) for s, i in zip(seqs, ids)])


def compute(seqs, **kwargs):
    """Classify a hand-built alignment and compute its flank spectra."""
    cls = classify_alignment(make_alignment(seqs), progress=False, **kwargs)
    return compute_flank_spectra(
        cls, sample_names=[f'seq{i}' for i in range(len(seqs))]
    )


# ---------------------------------------------------------------------------
# Channel bookkeeping
# ---------------------------------------------------------------------------


def test_channel_labels_count_and_uniqueness():
    """Each centre yields 16 unique 4 bp labels with the fixed centre."""
    ca = flank_channel_labels('CA')
    ta = flank_channel_labels('TA')
    assert len(ca) == 16 and len(set(ca)) == 16
    assert len(ta) == 16 and len(set(ta)) == 16
    assert all(lbl[1:3] == 'CA' for lbl in ca)
    assert all(lbl[1:3] == 'TA' for lbl in ta)


def test_channel_label_order():
    """Up base outer, down base inner: 'ACAA' first, 'GCAT' at 11, 'TCAT' last."""
    ca = flank_channel_labels('CA')
    assert ca[0] == 'ACAA'
    assert ca[GT_CHANNEL] == 'GCAT'
    assert ca[15] == 'TCAT'
    assert FLANK16_LABELS_CA[GT_CHANNEL] == 'GCAT'
    assert FLANK16_LABELS_TA[GT_CHANNEL] == 'GTAT'


def test_idx16_table_is_up_times_four_plus_down():
    """IDX16_TABLE[up, down] == up*4 + down for every flank pair."""
    for up in range(4):
        for down in range(4):
            assert IDX16_TABLE[up, down] == up * 4 + down


def test_comp_code_is_watson_crick():
    """COMP_CODE complements base codes: A<->T (0<->3), C<->G (1<->2)."""
    assert list(COMP_CODE) == [3, 2, 1, 0]


def test_flank_channel_labels_rejects_bad_center():
    """A non-two-base or non-ACGT centre is rejected."""
    with pytest.raises(ValueError):
        flank_channel_labels('C')
    with pytest.raises(ValueError):
        flank_channel_labels('CN')


# ---------------------------------------------------------------------------
# Substrate counting and folding
# ---------------------------------------------------------------------------


def test_forward_substrate_single_motif():
    """A single forward 'GCAT' motif lands in the G_T substrate channel."""
    result = compute(['GCAT', 'GCAT'])
    assert result.sub_fwd[GT_CHANNEL].tolist() == [1.0, 1.0]
    # No other channel is populated and there is no reverse substrate here.
    assert result.sub_fwd.sum() == 2.0
    assert result.sub_rev.sum() == 0.0


def test_reverse_substrate_folds_to_same_channel():
    """A physical 'ATGC' (reverse TpG) folds onto the same G_T channel."""
    result = compute(['ATGC', 'ATGC'])
    # tg at col2 (G), physical up=A down=C -> fold up=comp(C)=G, down=comp(A)=T.
    assert result.sub_rev[GT_CHANNEL].tolist() == [1.0, 1.0]
    assert result.sub_rev.sum() == 2.0
    assert result.sub_fwd.sum() == 0.0


def test_substrate_counted_anywhere_not_just_rip_columns():
    """Substrate CpA is counted even with no product anywhere in the column."""
    # No T in the alignment, so no product/RIP column exists, yet the CpA
    # substrate must still be counted.
    result = compute(['GCAG', 'GCAG'])
    # up=G, down=G -> channel 2*4 + 2 = 10.
    assert result.sub_fwd[10].tolist() == [1.0, 1.0]
    assert result.sub_fwd.sum() == 2.0


# ---------------------------------------------------------------------------
# Product counting and folding
# ---------------------------------------------------------------------------


def test_forward_product_single_motif():
    """A forward RIP column: 'GCAT' survivor + 'GTAT' product -> G_T product."""
    result = compute(['GCAT', 'GTAT'])
    # Row 0 is the surviving substrate, row 1 the product.
    assert result.sub_fwd[GT_CHANNEL].tolist() == [1.0, 0.0]
    assert result.prod_fwd[GT_CHANNEL].tolist() == [0.0, 1.0]
    assert result.prod_rev.sum() == 0.0


def test_reverse_product_folds_to_same_channel():
    """A reverse RIP column: 'ATGC' survivor + 'ATAC' product -> G_T product."""
    result = compute(['ATGC', 'ATAC'])
    assert result.sub_rev[GT_CHANNEL].tolist() == [1.0, 0.0]
    assert result.prod_rev[GT_CHANNEL].tolist() == [0.0, 1.0]
    assert result.prod_fwd.sum() == 0.0


# ---------------------------------------------------------------------------
# Gap-awareness and missing flanks
# ---------------------------------------------------------------------------


def test_gap_spanning_dinucleotide_and_flanks():
    """A CpA split by a gap column still resolves via nearest non-gap bases."""
    # G C - A T : C(col1) A(col3) is CpA across the gap; up=G(0) down=T(4).
    result = compute(['GC-AT', 'GC-AT'])
    assert result.sub_fwd[GT_CHANNEL].tolist() == [1.0, 1.0]


def test_missing_upstream_flank_is_skipped_and_tallied():
    """A CpA at the 5' edge has no upstream flank; it is dropped and counted."""
    result = compute(['CAT', 'CAT'])
    assert result.sub_fwd.sum() == 0.0
    assert result.n_skipped_flank['sub_fwd'] == 2


def test_missing_downstream_flank_is_skipped_and_tallied():
    """A CpA at the 3' edge has no downstream flank; it is dropped and counted."""
    result = compute(['GCA', 'GCA'])
    assert result.sub_fwd.sum() == 0.0
    assert result.n_skipped_flank['sub_fwd'] == 2


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------


def test_combined_equals_forward_plus_reverse():
    """substrate_combined/product_combined equal the per-strand sums."""
    result = compute(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    np.testing.assert_array_equal(
        result.substrate_combined(), result.sub_fwd + result.sub_rev
    )
    np.testing.assert_array_equal(
        result.product_combined(), result.prod_fwd + result.prod_rev
    )


def test_pooled_equals_row_sum():
    """pooled() sums each matrix across sequences."""
    result = compute(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    pooled = result.pooled()
    for key in ('sub_fwd', 'sub_rev', 'prod_fwd', 'prod_rev'):
        np.testing.assert_array_equal(pooled[key], getattr(result, key).sum(axis=1))


def test_matrix_selector():
    """matrix() returns the requested state/strand and rejects bad args."""
    result = compute(['GCAT', 'GTAT'])
    np.testing.assert_array_equal(result.matrix('substrate', 'forward'), result.sub_fwd)
    np.testing.assert_array_equal(
        result.matrix('product', 'combined'), result.product_combined()
    )
    with pytest.raises(ValueError):
        result.matrix('bogus', 'forward')
    with pytest.raises(ValueError):
        result.matrix('substrate', 'sideways')


def test_as_dict_roundtrips_shapes():
    """as_dict serialises the four matrices and metadata."""
    d = compute(['GCAT', 'GTAT']).as_dict()
    assert len(d['sub_fwd']) == 16 and len(d['sub_fwd'][0]) == 2
    assert d['channels_substrate'][GT_CHANNEL] == 'GCAT'
    assert set(d['n_skipped_flank']) == {'sub_fwd', 'sub_rev', 'prod_fwd', 'prod_rev'}


def test_sample_names_length_validation():
    """A mismatched sample_names length is rejected."""
    cls = classify_alignment(make_alignment(['GCAT', 'GCAT']), progress=False)
    with pytest.raises(ValueError):
        compute_flank_spectra(cls, sample_names=['only-one'])


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------


def test_compare_returns_five_comparisons():
    """compare_flank_spectra yields the five expected comparison keys."""
    result = compute(['GCAT', 'GTAT', 'ATGC', 'ATAC'])
    cmp = compare_flank_spectra(result, 0)
    assert set(cmp) == set(COMPARISON_KEYS)
    for value in cmp.values():
        assert 'cosine_similarity' in value
        assert 'chi2_reliable' in value
        assert 'n_a' in value and 'n_b' in value


def test_compare_identical_spectra_cosine_one():
    """Identical substrate/product profiles give cosine 1.0."""
    # A hand-made result via monkey-free construction: reuse compute output but
    # force product == substrate by comparing a matrix against itself.
    from derip2.stats.spectra_compare import compare_spectra

    vec = np.zeros(16)
    vec[GT_CHANNEL] = 5.0
    same = compare_spectra(vec, vec, channels=FLANK16_PAIR_LABELS)
    assert same['cosine_similarity'] == pytest.approx(1.0)


def test_compare_disjoint_spectra_cosine_zero():
    """Non-overlapping channels give cosine 0.0."""
    from derip2.stats.spectra_compare import compare_spectra

    a = np.zeros(16)
    b = np.zeros(16)
    a[0] = 3.0
    b[15] = 3.0
    assert compare_spectra(a, b)['cosine_similarity'] == pytest.approx(0.0)


def test_compare_all_zero_state_returns_nan_without_raising():
    """A state with no sites yields a nan p-value, not an exception."""
    # Only substrate present (no T anywhere -> no product), so product spectra
    # are all zero and the substrate-vs-product comparison degenerates cleanly.
    result = compute(['GCAG', 'GCAG'])
    cmp = compare_flank_spectra(result, 0)
    prod_cmp = cmp['sub_vs_prod_combined']
    assert np.isnan(prod_cmp['pvalue'])
    assert prod_cmp['chi2_reliable'] is False


def test_chi2_reliable_flag_respects_min_sites():
    """chi2_reliable is True only when both totals reach min_sites."""
    # Construct a one-sequence result with 25 substrate and 25 product sites so
    # the min-count threshold can be crossed either way.
    sub_fwd = np.zeros((16, 1))
    prod_fwd = np.zeros((16, 1))
    sub_fwd[GT_CHANNEL, 0] = 25.0
    prod_fwd[GT_CHANNEL, 0] = 25.0
    result = FlankSpectraResult(
        sub_fwd=sub_fwd,
        sub_rev=np.zeros((16, 1)),
        prod_fwd=prod_fwd,
        prod_rev=np.zeros((16, 1)),
        sample_names=['s'],
        n_skipped_flank=dict.fromkeys(
            ('sub_fwd', 'sub_rev', 'prod_fwd', 'prod_rev'), 0
        ),
    )
    reliable = compare_flank_spectra(result, 0, min_sites=20)
    unreliable = compare_flank_spectra(result, 0, min_sites=30)
    assert reliable['sub_vs_prod_combined']['chi2_reliable'] is True
    assert reliable['sub_vs_prod_combined']['n_a'] == 25.0
    assert unreliable['sub_vs_prod_combined']['chi2_reliable'] is False
