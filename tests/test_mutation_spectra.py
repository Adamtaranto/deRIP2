"""
Tests for the tree-free mutation-spectrum computation.

Synthetic alignments with hand-derived answers pin the event calling, context
resolution, homoplasy proxy and skip accounting. A golden regression over
``mintest.fa`` locks the assembled matrices end to end. Regenerate the golden
file after a reviewed behaviour change with::

    DERIP_REGEN=1 pytest tests/test_mutation_spectra.py
"""

import json
import logging
import os
from types import SimpleNamespace

from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import pytest

from derip2.aln_ops import alignment_to_array
from derip2.derip import DeRIP
from derip2.spectra.channels import DOWNSTREAM_INDEX, SBS96_INDEX, SBS192_INDEX
from derip2.stats import compute_spectra
from derip2.stats.mutation_spectra import SpectraResult

logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
GOLDEN_DIR = os.path.join(HERE, 'data', 'golden')
MINTEST = os.path.join(HERE, 'data', 'mintest.fa')


def _classes(seqs):
    """Wrap sequence strings as an object exposing ``arr`` like a classification."""
    aln = MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=f'seq{i}') for i, s in enumerate(seqs)]
    )
    return SimpleNamespace(arr=alignment_to_array(aln))


def test_single_event_channel():
    """One planted C>T in A_G context lands in the A[C>T]G channel."""
    ancestor = 'ACGTACGT'
    cls = _classes(['ACGTACGT', 'ATGTACGT'])
    res = compute_spectra(cls, ancestor)
    assert res.event_rows.size == 1
    assert res.sbs96[SBS96_INDEX['A[C>T]G'], 0] == 1
    assert res.sbs192[SBS192_INDEX['A[C>T]G'], 0] == 1
    assert res.n_indel_or_ambiguous == 0
    assert res.n_unassignable_context == 0


def test_purine_event_folds_into_sbs96_but_not_sbs192_pyrimidine():
    """A G>A event folds to a pyrimidine SBS-96 channel, stays G>A in SBS-192."""
    ancestor = 'ACGTA'
    # Row changes col 2 (ancestor G) to A: G>A, 5'=C, 3'=T.
    cls = _classes(['ACGTA', 'ACATA'])
    res = compute_spectra(cls, ancestor)
    # SBS-96 folds G>A in C_T to A[C>T]G.
    assert res.sbs96[SBS96_INDEX['A[C>T]G'], 0] == 1
    # SBS-192 keeps the purine identity.
    assert res.sbs192[SBS192_INDEX['C[G>A]T'], 0] == 1


def test_homoplasy_counts_independent_hits():
    """The same substitution on two rows registers two independent hits."""
    ancestor = 'ACGTA'
    cls = _classes(['ACGTA', 'ATGTA', 'ATGTA'])  # col 1 C>T on rows 1 and 2
    res = compute_spectra(cls, ancestor)
    table = res.homoplasy_table(min_hits=2)
    assert len(table) == 1
    assert table[0] == {'col': 1, 'ref': 'C', 'alt': 'T', 'n_independent': 2}
    # A single hit is not reported at the >= 2 threshold.
    assert res.homoplasy_table(min_hits=3) == []


def test_indel_is_not_a_substitution():
    """A gap opposite an ancestral base is skipped as an indel, not an SBS."""
    ancestor = 'ACGT'
    cls = _classes(['ACGT', 'A-GT'])
    res = compute_spectra(cls, ancestor)
    assert res.event_rows.size == 0
    assert res.n_indel_or_ambiguous == 1


def test_terminal_column_is_unassignable():
    """A substitution in a terminal column has no full context and is dropped."""
    ancestor = 'CAT'
    cls = _classes(['CAT', 'TAT'])  # col 0 C>T, no 5' flank
    res = compute_spectra(cls, ancestor)
    assert res.event_rows.size == 0
    assert res.n_unassignable_context == 1
    assert res.sbs96.sum() == 0


def test_ambiguous_ancestor_base_skipped():
    """An N in the ancestor cannot anchor a substitution."""
    ancestor = 'ANGT'
    cls = _classes(['ANGT', 'ACGT'])  # col 1 ancestor N vs observed C
    res = compute_spectra(cls, ancestor)
    assert res.event_rows.size == 0
    assert res.n_indel_or_ambiguous == 1


def test_per_row_samples_sum_to_pooled():
    """Splitting by row and pooling give the same total per channel."""
    cls = _classes(['ACGTACGT', 'ATGTACGT', 'ACGTATGT'])
    pooled = compute_spectra(cls, 'ACGTACGT')
    per_row = compute_spectra(cls, 'ACGTACGT', samples=['a', 'b', 'c'])
    assert per_row.sbs96.shape == (96, 3)
    assert np.array_equal(per_row.sbs96.sum(axis=1), pooled.sbs96[:, 0])


def test_length_mismatch_raises():
    """An ancestor of the wrong length is rejected."""
    cls = _classes(['ACGT', 'ACGT'])
    with pytest.raises(ValueError):
        compute_spectra(cls, 'ACG')


def test_mintest_pipeline_totals():
    """The SBS-96 and SBS-192 totals agree and match the event count."""
    d = DeRIP(MINTEST)
    d.calculate_rip()
    res = d.calculate_spectra()
    assert isinstance(res, SpectraResult)
    total_events = res.event_rows.size
    assert res.sbs96.sum() == total_events
    assert res.sbs192.sum() == total_events
    # mintest is a RIP-like alignment: C>T should dominate the spectrum.
    ct_total = sum(
        res.sbs96[SBS96_INDEX[f'{a}[C>T]{b}'], 0] for a in 'ACGT' for b in 'ACGT'
    )
    assert ct_total > 0
    assert ct_total == res.sbs96[32:48, 0].sum()  # C>T block is rows 32..47


def test_mintest_matches_golden():
    """The assembled mintest spectra match the committed golden reference."""
    d = DeRIP(MINTEST)
    d.calculate_rip()
    result = d.calculate_spectra().as_dict()
    golden_path = os.path.join(GOLDEN_DIR, 'mintest_spectra.json')

    if os.environ.get('DERIP_REGEN') or not os.path.exists(golden_path):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(golden_path, 'w') as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        pytest.skip(f'Regenerated golden file: {golden_path}')

    with open(golden_path) as fh:
        golden = json.load(fh)
    result = json.loads(json.dumps(result))
    assert result == golden


# ---------------------------------------------------------------------------
# Downstream-triplet context.
# ---------------------------------------------------------------------------

# Reverse-complement translation preserving gaps, for orientation-invariance.
_RC_TABLE = str.maketrans('ACGTacgt-', 'TGCAtgca-')


def _rc(seq):
    """Reverse-complement a gapped sequence string."""
    return seq.translate(_RC_TABLE)[::-1]


def test_invalid_context_raises():
    """An unknown context is rejected."""
    with pytest.raises(ValueError):
        compute_spectra(_classes(['ACGT', 'ACGT']), 'ACGT', context='bogus')


def test_downstream_single_event_channel():
    """A C>T with downstream A then G lands in the [C>T]AG downstream channel."""
    ancestor = 'TCAGT'
    cls = _classes(['TCAGT', 'TTAGT'])  # col 1 C>T; downstream A (col2), G (col3)
    res = compute_spectra(cls, ancestor, context='downstream')
    assert res.context == 'downstream'
    assert res.sbs192 is None
    assert res.sbs96.shape == (96, 1)
    assert res.event_rows.size == 1
    assert res.sbs96[DOWNSTREAM_INDEX['[C>T]AG'], 0] == 1


def test_downstream_purine_event_uses_complemented_upstream():
    """A G>A event folds to a C>T downstream channel via complemented upstream."""
    # col 3 ancestor G (purine); upstream bases A (col2) then C (col1). Their
    # complements T, G are the pyrimidine-strand downstream bases -> [C>T]TG.
    ancestor = 'TCAGT'
    cls = _classes(['TCAGT', 'TCAAT'])  # col 3 G>A
    res = compute_spectra(cls, ancestor, context='downstream')
    assert res.event_rows.size == 1
    assert res.sbs96[DOWNSTREAM_INDEX['[C>T]TG'], 0] == 1


def test_downstream_end_of_sequence_is_unassignable():
    """A pyrimidine ref lacking a second downstream base is unassignable."""
    ancestor = 'ACT'
    cls = _classes(['ACT', 'ATT'])  # col 1 C>T but only one downstream base
    res = compute_spectra(cls, ancestor, context='downstream')
    assert res.event_rows.size == 0
    assert res.n_unassignable_context == 1
    assert res.sbs96.sum() == 0


def test_downstream_orientation_invariance():
    """The downstream matrix is identical whichever strand the input is given in.

    This is the load-bearing correctness property: reverse-complementing the
    ancestor and every row (and reversing the column order) must not change any
    downstream channel count.
    """
    anc = 'ACGTACGTAC'
    seqs = ['ACGTACGTAC', 'ACGTATGTAC', 'ATGTACGTGC']
    forward = compute_spectra(_classes(seqs), anc, context='downstream')
    reverse = compute_spectra(
        _classes([_rc(s) for s in seqs]), _rc(anc), context='downstream'
    )
    assert forward.sbs96.sum() > 0
    assert np.array_equal(forward.sbs96, reverse.sbs96)


def test_downstream_matches_golden():
    """The assembled mintest downstream spectra match the committed golden."""
    d = DeRIP(MINTEST)
    d.calculate_rip()
    result = compute_spectra(
        d.column_classes, str(d.gapped_consensus.seq), context='downstream'
    ).as_dict()
    golden_path = os.path.join(GOLDEN_DIR, 'mintest_spectra_downstream.json')

    if os.environ.get('DERIP_REGEN') or not os.path.exists(golden_path):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(golden_path, 'w') as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        pytest.skip(f'Regenerated golden file: {golden_path}')

    with open(golden_path) as fh:
        golden = json.load(fh)
    result = json.loads(json.dumps(result))
    assert result == golden
