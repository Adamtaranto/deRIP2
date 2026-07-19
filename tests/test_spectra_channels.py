"""
Unit tests for SBS-96 / SBS-192 channel bookkeeping and context resolution.

These pin the correctness-critical bits of the mutation-spectrum feature: the
pyrimidine folding (flanks must be reverse-complemented *and* swapped), the
canonical channel orderings, and the nearest-non-gap trinucleotide context
lookup including its terminal-column edge case.
"""

import itertools
import logging

import numpy as np

from derip2.spectra.channels import (
    BASES,
    DOWNSTREAM_CHANNELS,
    DOWNSTREAM_INDEX,
    SBS96_CHANNELS,
    SBS96_INDEX,
    SBS192_CHANNELS,
    SBS192_INDEX,
    downstream_channel,
    downstream_context,
    fold_to_pyrimidine,
    parse_downstream_channel,
    sbs96_channel,
    sbs192_channel,
    trinucleotide_context,
)

logging.disable(logging.CRITICAL)


def test_channel_counts_and_uniqueness():
    """There are exactly 96 and 192 unique channels in canonical order."""
    assert len(SBS96_CHANNELS) == 96
    assert len(set(SBS96_CHANNELS)) == 96
    assert len(SBS192_CHANNELS) == 192
    assert len(set(SBS192_CHANNELS)) == 192
    # The index maps back to the ordered position.
    assert SBS96_INDEX[SBS96_CHANNELS[0]] == 0
    assert SBS96_INDEX[SBS96_CHANNELS[-1]] == 95


def test_all_sbs96_channels_are_pyrimidine():
    """Every SBS-96 channel's reference base is a pyrimidine (C or T)."""
    refs = {label[2] for label in SBS96_CHANNELS}
    assert refs == {'C', 'T'}


def test_folding_of_known_purine_event():
    """G>A in T_C folds to C>T in G_A (flanks swapped and complemented)."""
    assert fold_to_pyrimidine('T', 'G', 'A', 'C') == ('G', 'C', 'T', 'A')
    assert sbs96_channel('T', 'G', 'A', 'C') == 'G[C>T]A'


def test_pyrimidine_event_unchanged_by_folding():
    """A pyrimidine-reference event is returned unchanged."""
    assert fold_to_pyrimidine('A', 'C', 'T', 'G') == ('A', 'C', 'T', 'G')
    assert sbs96_channel('A', 'C', 'T', 'G') == 'A[C>T]G'


def test_sbs192_keeps_purine_reference():
    """SBS-192 does not fold: a purine-reference event keeps its identity."""
    assert sbs192_channel('A', 'G', 'T', 'A') == 'A[G>T]A'
    assert 'A[G>T]A' in SBS192_INDEX


def test_every_event_maps_to_a_channel():
    """Every non-identity substitution maps into both channel sets."""
    for five, ref, alt, three in itertools.product(BASES, BASES, BASES, BASES):
        if ref == alt:
            continue
        assert sbs96_channel(five, ref, alt, three) in SBS96_INDEX
        assert sbs192_channel(five, ref, alt, three) in SBS192_INDEX


def test_folding_is_reversible_partner():
    """A purine event and its pyrimidine complement fold to the same channel."""
    # C>T in A_G and its reverse complement G>A in C_T are the same SBS-96 event.
    assert sbs96_channel('A', 'C', 'T', 'G') == sbs96_channel('C', 'G', 'A', 'T')


def _neighbors(seq: str):
    """Compute (next_idx, prev_idx) for a 1-D sequence, treating '-' as gaps."""
    from derip2.aln_ops import _nongap_neighbors

    arr = np.frombuffer(seq.encode('ascii'), dtype='S1').reshape(1, -1)
    nxt, prv = _nongap_neighbors(arr)
    return arr[0], nxt[0], prv[0]


def test_context_skips_gaps():
    """The flanks are the nearest non-gap bases, skipping over gap columns."""
    seq, nxt, prv = _neighbors('A--C--T')
    # Column 3 is the C; nearest left non-gap is A (col 0), nearest right is T.
    assert trinucleotide_context(seq, 3, nxt, prv) == ('A', 'T')


def test_context_terminal_column_is_unassignable():
    """A mutation with no base on one side returns None (unassignable)."""
    seq, nxt, prv = _neighbors('C--A')
    # Column 0 has no non-gap base to its left.
    assert trinucleotide_context(seq, 0, nxt, prv) is None
    # Column 3 (the A) has no non-gap base to its right.
    assert trinucleotide_context(seq, 3, nxt, prv) is None


def test_context_adjacent_bases():
    """With no gaps the flanks are simply the neighbouring columns."""
    seq, nxt, prv = _neighbors('ACGTA')
    assert trinucleotide_context(seq, 2, nxt, prv) == ('C', 'T')


# ---------------------------------------------------------------------------
# Downstream-triplet context.
# ---------------------------------------------------------------------------


def test_downstream_channel_counts_and_uniqueness():
    """There are exactly 96 unique downstream channels with a bijective index."""
    assert len(DOWNSTREAM_CHANNELS) == 96
    assert len(set(DOWNSTREAM_CHANNELS)) == 96
    assert DOWNSTREAM_INDEX[DOWNSTREAM_CHANNELS[0]] == 0
    assert DOWNSTREAM_INDEX[DOWNSTREAM_CHANNELS[-1]] == 95


def test_downstream_channels_are_pyrimidine():
    """Every downstream channel's reference base is a pyrimidine (C or T)."""
    refs = {parse_downstream_channel(label)[0] for label in DOWNSTREAM_CHANNELS}
    assert refs == {'C', 'T'}


def test_downstream_label_form_is_distinct_from_sbs96():
    """Downstream labels start with '[' so an SBS-96 parser cannot mistake them."""
    assert downstream_channel('C', 'T', 'A', 'G') == '[C>T]AG'
    assert all(label.startswith('[') for label in DOWNSTREAM_CHANNELS)


def test_parse_downstream_channel_round_trip():
    """Every canonical label parses back to its (ref, alt, d1, d2) components."""
    for label in DOWNSTREAM_CHANNELS:
        ref, alt, d1, d2 = parse_downstream_channel(label)
        assert downstream_channel(ref, alt, d1, d2) == label


def test_parse_downstream_channel_rejects_sbs96():
    """A trinucleotide (SBS-96) label is not a valid downstream label."""
    import pytest

    with pytest.raises(ValueError):
        parse_downstream_channel('A[C>T]G')


def test_downstream_context_pyrimidine_native():
    """A pyrimidine reference reads its two bases directly downstream."""
    seq, nxt, prv = _neighbors('ACGTA')
    # Column 1 is C (pyrimidine); the two downstream bases are G then T.
    assert downstream_context(seq, 1, nxt, prv) == ('G', 'T')


def test_downstream_context_skips_gaps_by_chaining():
    """The two downstream bases are the nearest non-gap neighbours, chained."""
    seq, nxt, prv = _neighbors('C--G--T')
    # Column 0 is C; nearest downstream non-gaps are G (col 3) then T (col 6).
    assert downstream_context(seq, 0, nxt, prv) == ('G', 'T')


def test_downstream_context_purine_uses_complemented_upstream():
    """A purine reference reads the reverse-complement of its two upstream bases."""
    seq, nxt, prv = _neighbors('TCAG')
    # Column 3 is G (purine); upstream bases are A (col 2) then C (col 1).
    # Their complements T and G are the pyrimidine-strand downstream bases.
    assert downstream_context(seq, 3, nxt, prv) == ('T', 'G')


def test_downstream_context_none_when_second_downstream_missing():
    """A pyrimidine reference with only one downstream base is unassignable."""
    seq, nxt, prv = _neighbors('ACT')
    # Column 1 is C; there is a downstream T but no second downstream base.
    assert downstream_context(seq, 1, nxt, prv) is None


def test_downstream_context_none_for_purine_near_five_prime_end():
    """A purine reference too close to the 5' end has no full upstream context."""
    seq, nxt, prv = _neighbors('AGCT')
    # Column 1 is G (purine) but only one upstream base (A) exists.
    assert downstream_context(seq, 1, nxt, prv) is None
