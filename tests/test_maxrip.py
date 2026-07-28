"""
Tests for the maximum-RIP counterfactual sequences.

These use small hand-built alignments so each rule can be isolated: which sites a
variant converts, how gaps and sequence ends are handled, and the invariants that
must hold for every variant (idempotence, gap preservation, the subset relation
between the three rules). A run on the real ``mintest`` fixture checks the same
invariants against biological data.
"""

from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pytest

from derip2.derip import DeRIP
from derip2.maxrip import (
    MAX_RIP_VARIANTS,
    compute_max_rip,
    max_rip_multifasta,
    write_max_rip_fasta,
)


def build_alignment(*sequences):
    """
    Build a MultipleSeqAlignment from raw sequence strings.

    Parameters
    ----------
    *sequences : str
        Equal-length sequences, one per row.

    Returns
    -------
    Bio.Align.MultipleSeqAlignment
        The alignment, with rows named ``r0``, ``r1``, ...
    """
    return MultipleSeqAlignment(
        [SeqRecord(Seq(seq), id=f'r{i}') for i, seq in enumerate(sequences)]
    )


def run_derip(*sequences, **kwargs):
    """
    Build an alignment and run the deRIP pipeline over it.

    Parameters
    ----------
    *sequences : str
        Equal-length sequences, one per row.
    **kwargs
        Forwarded to :class:`derip2.derip.DeRIP`.

    Returns
    -------
    derip2.derip.DeRIP
        The instance, with ``calculate_rip`` already run.
    """
    derip = DeRIP(build_alignment(*sequences), **kwargs)
    derip.calculate_rip()
    return derip


# -- per-variant rules ---------------------------------------------------------


def test_forward_substrate_sites_convert():
    """Every CpA in the consensus becomes TpA under the 'all' variant."""
    derip = run_derip('CAACAA', 'TAACAA')
    assert derip.get_consensus_string() == 'CAACAA'

    result = derip.calculate_max_rip('all')
    assert result.seq == 'TAATAA'
    assert result.converted_cols.tolist() == [0, 3]
    assert result.strand.tolist() == [b'+', b'+']
    assert (result.n_forward, result.n_reverse) == (2, 0)


def test_reverse_substrate_sites_convert():
    """Every TpG in the consensus becomes TpA under the 'all' variant."""
    derip = run_derip('TGTTGT', 'TATTGT')
    assert derip.get_consensus_string() == 'TGTTGT'

    result = derip.calculate_max_rip('all')
    assert result.seq == 'TATTAT'
    assert result.converted_cols.tolist() == [1, 4]
    assert result.strand.tolist() == [b'-', b'-']
    assert (result.n_forward, result.n_reverse) == (0, 2)


def test_substrate_site_split_by_a_gap_converts():
    """A CpA interrupted by a gap column is still a substrate site.

    Gap-skipping matches the alignment scan, so ``C-A`` behaves exactly as ``CA``
    does; the gap itself is preserved and shifts the ungapped offset.
    """
    derip = run_derip('ATC-AGGCG', 'ATC-AGGCG')
    assert str(derip.gapped_consensus.seq) == 'ATC-AGGCG'

    result = derip.calculate_max_rip('all')
    assert result.gapped_seq == 'ATT-AGGCG'
    assert result.seq == 'ATTAGGCG'
    # Column 2 in the alignment is offset 2 in the ungapped sequence (the gap
    # follows it), and the trailing G has a C before it, so it is not a TpG.
    assert result.converted_cols.tolist() == [2]
    assert result.converted_positions.tolist() == [2]


def test_terminal_bases_without_a_partner_are_left_alone():
    """A C at the 3' end and a G at the 5' end have no dinucleotide partner."""
    # Leading G (no 5' neighbour) and trailing C (no 3' neighbour); the interior
    # is deliberately free of CpA and TpG.
    derip = run_derip('GTTGGC', 'GTTGGC')
    assert derip.get_consensus_string() == 'GTTGGC'

    result = derip.calculate_max_rip('all')
    # Only the internal TpG at column 3 qualifies; the flanking G and C do not.
    assert result.converted_cols.tolist() == [3]
    assert result.seq == 'GTTAGC'


def test_observed_variant_skips_columns_without_rip_evidence():
    """'observed' converts only where the alignment shows RIP at that column.

    Column 0 varies C/T across rows, so RIP is evidenced there; column 3 is an
    invariant C with no product anywhere, so 'observed' leaves it while 'all'
    converts it.
    """
    derip = run_derip('CAACAA', 'TAACAA')

    observed = derip.calculate_max_rip('observed')
    every = derip.calculate_max_rip('all')

    assert observed.converted_cols.tolist() == [0]
    assert every.converted_cols.tolist() == [0, 3]
    assert observed.seq == 'TAACAA'


def test_nonrip_variant_converts_deamination_outside_rip_context():
    """'all_plus_nonrip' adds columns whose C to T change is not in CpA context.

    Column 1 is a C/T polymorphism followed by G, so it is deamination outside
    RIP dinucleotide context: only the 'all_plus_nonrip' variant touches it.
    """
    derip = run_derip('ACGATCAAT', 'ATGATCAAT', 'ACGATCAAT', 'ATGATTAAT')
    assert derip.get_consensus_string() == 'ACGATCAAT'

    every = derip.calculate_max_rip('all')
    plus = derip.calculate_max_rip('all_plus_nonrip')

    assert 1 not in every.converted_cols.tolist()
    assert 1 in plus.converted_cols.tolist()
    # The knock-on conversion of column 2 is covered by the cascade test below.
    assert plus.seq[1] == 'T'


def test_context_conversions_do_not_cascade():
    """A context-derived conversion never creates a new target.

    For 'all' and 'observed' this makes a single pass exact: converting a CpA
    cannot produce a TpG, nor a TpG a CpA. Checked by confirming that the sites
    found are exactly those present in the original consensus.
    """
    derip = run_derip('CATGCAACATG', 'TATGCAACATG')
    consensus = str(derip.gapped_consensus.seq)

    for variant in ('all', 'observed'):
        result = derip.calculate_max_rip(variant)
        for col, strand in zip(result.converted_cols, result.strand):
            # The partner base is read from the *unconverted* consensus, so a
            # site only qualified because of context that already existed.
            if strand == b'+':
                assert consensus[col + 1] == 'A'
            else:
                assert consensus[col - 1] == 'T'


def test_nonrip_conversion_cascades_to_a_fixed_point():
    """A context-free deamination can expose a new RIP substrate.

    Converting a C to T immediately 5' of a G creates a TpG that was not a
    substrate in the original consensus, so 'all_plus_nonrip' has to keep going
    until nothing changes. Column 1 here is a non-RIP C/T column followed by a G.
    """
    derip = run_derip('ACGATCAAT', 'ATGATCAAT', 'ACGATCAAT', 'ATGATTAAT')
    assert derip.get_consensus_string() == 'ACGATCAAT'

    result = derip.calculate_max_rip('all_plus_nonrip')
    # Column 1 (C->T, context-free) then exposes column 2 (G preceded by the new
    # T), which a single pass would have missed.
    assert result.converted_cols.tolist() == [1, 2, 5]
    assert result.seq == 'ATAATTAAT'
    assert result.strand.tolist() == [b'+', b'-', b'+']


@pytest.mark.parametrize('variant', MAX_RIP_VARIANTS)
def test_result_is_a_fixed_point(variant):
    """Re-running any variant on its own output converts nothing further."""
    derip = run_derip('ACGATCAAT', 'ATGATCAAT', 'ACGATCAAT', 'ATGATTAAT')
    result = derip.calculate_max_rip(variant)
    again = compute_max_rip(result.gapped_seq, derip.column_classes, variant=variant)
    assert again.converted_cols.size == 0
    assert again.gapped_seq == result.gapped_seq


# -- invariants that hold for every variant ------------------------------------


@pytest.fixture
def invariant_derip():
    """
    A small alignment exercising forward, reverse, gap and non-RIP cases.

    Returns
    -------
    derip2.derip.DeRIP
        The instance, with ``calculate_rip`` already run.
    """
    return run_derip(
        'CAACATG-CGATCAA',
        'TAACATG-CGATCAA',
        'CAACATG-TGATTAA',
        'CAATATG-CGATCAA',
    )


@pytest.mark.parametrize('variant', MAX_RIP_VARIANTS)
def test_structural_invariants(invariant_derip, variant):
    """Gaps, length and untouched columns are preserved by every variant."""
    consensus = str(invariant_derip.gapped_consensus.seq)
    result = invariant_derip.calculate_max_rip(variant)

    assert len(result.gapped_seq) == invariant_derip.alignment.get_alignment_length()
    assert result.seq == result.gapped_seq.replace('-', '')
    # Gap columns are untouched and stay in place.
    assert [i for i, c in enumerate(result.gapped_seq) if c == '-'] == [
        i for i, c in enumerate(consensus) if c == '-'
    ]
    # Only the reported columns differ from the deRIP consensus.
    changed = {
        i for i, (a, b) in enumerate(zip(consensus, result.gapped_seq)) if a != b
    }
    assert changed == set(result.converted_cols.tolist())


@pytest.mark.parametrize('variant', MAX_RIP_VARIANTS)
def test_conversions_are_cytosine_or_guanine_to_product(invariant_derip, variant):
    """Every converted site was a C or G and became the matching product base."""
    consensus = str(invariant_derip.gapped_consensus.seq)
    result = invariant_derip.calculate_max_rip(variant)

    for col, strand in zip(result.converted_cols, result.strand):
        if strand == b'+':
            assert consensus[col] == 'C'
            assert result.gapped_seq[col] == 'T'
        else:
            assert consensus[col] == 'G'
            assert result.gapped_seq[col] == 'A'

    # The ungapped offsets index the same bases in the ungapped sequence.
    for col, pos in zip(result.converted_cols, result.converted_positions):
        assert result.seq[pos] == result.gapped_seq[col]


@pytest.mark.parametrize('variant', MAX_RIP_VARIANTS)
def test_idempotent(invariant_derip, variant):
    """Applying a variant to its own output changes nothing."""
    result = invariant_derip.calculate_max_rip(variant)
    again = compute_max_rip(
        result.gapped_seq, invariant_derip.column_classes, variant=variant
    )
    assert again.converted_cols.size == 0


def test_variants_are_nested(invariant_derip):
    """observed is a subset of all, which is a subset of all_plus_nonrip."""
    sites = {
        variant: set(invariant_derip.calculate_max_rip(variant).converted_cols.tolist())
        for variant in MAX_RIP_VARIANTS
    }
    assert sites['observed'] <= sites['all'] <= sites['all_plus_nonrip']


def test_invariants_hold_on_real_alignment(mintest_path):
    """The same invariants hold on the biological mintest fixture."""
    derip = DeRIP(mintest_path)
    derip.calculate_rip()
    consensus = str(derip.gapped_consensus.seq)

    sites = {}
    for variant in MAX_RIP_VARIANTS:
        result = derip.calculate_max_rip(variant)
        sites[variant] = set(result.converted_cols.tolist())
        assert len(result.gapped_seq) == len(consensus)
        assert result.seq == result.gapped_seq.replace('-', '')
        assert all(consensus[col] in 'CG' for col in result.converted_cols)
        assert all(result.gapped_seq[col] in 'TA' for col in result.converted_cols)

    assert sites['observed'] <= sites['all'] <= sites['all_plus_nonrip']
    # The default variant should find real substrate to convert in this fixture.
    assert sites['all']


# -- edge cases and errors -----------------------------------------------------


def test_all_gap_column_is_never_converted():
    """A column that is all gaps stays a gap and is not a dinucleotide partner."""
    # Column 2 is a gap in every row, so the consensus carries a gap there.
    derip = run_derip('CA--CAAT', 'CA--CAAT')
    result = derip.calculate_max_rip('all')
    assert result.gapped_seq[2] == '-'
    assert 2 not in result.converted_cols.tolist()


def test_unknown_variant_raises():
    """An unrecognised variant name is rejected with the valid options listed."""
    derip = run_derip('CAACAA', 'TAACAA')
    with pytest.raises(ValueError, match='Unknown maximum-RIP variant'):
        derip.calculate_max_rip('everything')


def test_length_mismatch_raises():
    """A consensus that is not the alignment width is rejected."""
    derip = run_derip('CAACAA', 'TAACAA')
    with pytest.raises(ValueError, match='does not match the alignment width'):
        compute_max_rip('CAA', derip.column_classes, variant='all')


def test_requires_calculate_rip_first():
    """The DeRIP accessor refuses to run before the alignment is classified."""
    derip = DeRIP(build_alignment('CAACAA', 'TAACAA'))
    with pytest.raises(ValueError, match='Must call calculate_rip'):
        derip.calculate_max_rip('all')


# -- DeRIP integration ---------------------------------------------------------


def test_results_are_cached_per_variant():
    """Repeated calls reuse the cached result; different variants do not share."""
    derip = run_derip('CAACAA', 'TAACAA')
    first = derip.calculate_max_rip('all')
    assert derip.calculate_max_rip('all') is first
    assert derip.calculate_max_rip('observed') is not first
    assert set(derip.max_rip_results) == {'all', 'observed'}


def test_cache_is_cleared_when_results_are_invalidated():
    """Replacing the alignment discards cached maximum-RIP sequences."""
    derip = run_derip('CAACAA', 'TAACAA')
    derip.calculate_max_rip('all')
    assert derip.max_rip_results

    derip._invalidate_results()
    assert derip.max_rip_results == {}


def test_accessors_agree_with_the_result_object():
    """The string/position accessors are views onto the same computation."""
    derip = run_derip('CAACAA', 'TAACAA')
    result = derip.calculate_max_rip('all')

    assert derip.get_max_rip_string('all') == result.seq
    assert derip.get_max_rip_string('all', gapped=True) == result.gapped_seq
    assert derip.get_max_rip_positions('all') == result.converted_positions.tolist()
    assert (
        derip.get_max_rip_positions('all', gapped=True)
        == result.converted_cols.tolist()
    )


def test_write_max_rip_round_trips(tmp_path):
    """Written records read back with the same sequences, one per variant."""
    from Bio import SeqIO

    derip = run_derip('CAACAA', 'TAACAA')
    out = tmp_path / 'maxrip.fasta'
    derip.write_max_rip(str(out))

    records = list(SeqIO.parse(str(out), 'fasta'))
    assert [rec.id for rec in records] == [
        f'maxRIPseq_{variant}' for variant in MAX_RIP_VARIANTS
    ]
    for record, variant in zip(records, MAX_RIP_VARIANTS):
        assert str(record.seq) == derip.get_max_rip_string(variant)


def test_write_max_rip_fasta_can_emit_gapped_sequences(tmp_path):
    """The gapped flag writes column-aligned sequences instead."""
    from Bio import SeqIO

    derip = run_derip('CA-ACAA', 'TA-ACAA')
    results = [derip.calculate_max_rip(v) for v in MAX_RIP_VARIANTS]
    out = tmp_path / 'maxrip_gapped.fasta'
    write_max_rip_fasta(results, str(out), gapped=True)

    records = list(SeqIO.parse(str(out), 'fasta'))
    assert all('-' in str(record.seq) for record in records)


def test_multifasta_string_is_plain_and_labelled():
    """The embedded-download rendering is plain FASTA naming each variant."""
    derip = run_derip('CAACAA', 'TAACAA')
    results = [derip.calculate_max_rip(v) for v in MAX_RIP_VARIANTS]
    text = max_rip_multifasta(results)

    assert text.count('>') == len(MAX_RIP_VARIANTS)
    assert '<' not in text.replace('>', '')
    for variant in MAX_RIP_VARIANTS:
        assert f'>maxRIPseq_{variant} ' in text
