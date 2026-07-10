"""
Tests for the vectorised RIP column classifier and the RSI statistic.

The first section fuzzes ``classify_columns``/``apply_classification`` against a
transcription of the original per-column scan. That reference implementation is
the contract: if the two ever disagree, the vectorised classifier has drifted
from the RIP-detection rules the consensus correction relies on.
"""

import logging

from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import pytest

from derip2.aln_ops import (
    _array_to_alignment,
    _nongap_neighbors,
    alignment_to_array,
    apply_classification,
    classify_alignment,
    classify_columns,
    initRIPCounter,
    initTracker,
    updateRIPCount,
    updateTracker,
)
from derip2.stats import compute_rsi
from derip2.stats.strand_bias import AMBIGUITY_POLICIES, SUBSTRATE_SCOPES

logging.disable(logging.CRITICAL)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def make_alignment(seqs):
    """Build a MultipleSeqAlignment from a list of sequence strings."""
    return MultipleSeqAlignment(
        [SeqRecord(Seq(s), id=f'seq{i}') for i, s in enumerate(seqs)]
    )


def _reference_correctRIP(
    align, tracker, RIPcounts, max_snp_noise, min_rip_like, reaminate
):
    """
    Transcription of the original per-column RIP scan, kept as a test oracle.

    This mirrors the pre-vectorisation implementation of ``correctRIP`` exactly,
    including its branch structure. Do not "clean it up" — its value is that it
    is an independent statement of the same rules.
    """
    from copy import deepcopy

    tracker = deepcopy(tracker)
    RIPcounts = deepcopy(RIPcounts)

    arr = alignment_to_array(align)
    n_rows, n_cols = arr.shape
    maskedArr = arr.copy()
    next_idx, prev_idx = _nongap_neighbors(arr)
    row_ids = np.arange(n_rows)

    bA, bT, bG, bC = b'A', b'T', b'G', b'C'
    corrected_positions = []
    add_fwd = np.zeros(n_rows, dtype=np.int64)
    add_rev = np.zeros(n_rows, dtype=np.int64)
    add_nonrip = np.zeros(n_rows, dtype=np.int64)

    markup = {
        'rip_product': set(),
        'rip_substrate': set(),
        'non_rip_deamination': set(),
    }

    def _mark(category, col, base, rows, offsets):
        for r, off in zip(rows.tolist(), offsets):
            markup[category].add((int(col), int(r), base, int(off)))

    for colIdx in range(n_cols):
        col = arr[:, colIdx]
        is_C, is_T, is_G, is_A = col == bC, col == bT, col == bG, col == bA
        baseCount = int(is_C.sum() + is_T.sum() + is_G.sum() + is_A.sum())
        if not baseCount:
            continue

        modC = modG = False
        fwd_TArows = rev_TArows = None

        CT_count = int(is_C.sum() + is_T.sum())
        GA_count = int(is_G.sum() + is_A.sum())
        CTprop = CT_count / baseCount
        GAprop = GA_count / baseCount

        nxt, prv = next_idx[:, colIdx], prev_idx[:, colIdx]
        has_next, has_prev = nxt >= 0, prv >= 0
        next_base = arr[row_ids, np.where(has_next, nxt, 0)]
        prev_base = arr[row_ids, np.where(has_prev, prv, 0)]

        if CTprop >= max_snp_noise:
            ca_rows = np.where(is_C & has_next & (next_base == bA))[0]
            _mark('rip_substrate', colIdx, 'C', ca_rows, nxt[ca_rows] - colIdx)

            if CTprop > GAprop and is_C.any() and is_T.any():
                ta_rows = np.where(is_T & has_next & (next_base == bA))[0]
                fwd_TArows = ta_rows
                TinCol = np.where(is_T)[0]

                if ca_rows.size and ta_rows.size:
                    propRIPlike = (ta_rows.size + ca_rows.size) / CT_count
                    add_fwd[ta_rows] += 1
                    _mark('rip_product', colIdx, 'T', ta_rows, nxt[ta_rows] - colIdx)
                    nonrip_T = np.setdiff1d(TinCol, ta_rows, assume_unique=True)
                    add_nonrip[nonrip_T] += 1
                    _mark(
                        'non_rip_deamination',
                        colIdx,
                        'T',
                        nonrip_T,
                        [0] * nonrip_T.size,
                    )
                    if propRIPlike >= min_rip_like or reaminate:
                        tracker = updateTracker(colIdx, 'C', tracker, force=False)
                        modC = True
                        corrected_positions.append(colIdx)
                else:
                    if reaminate:
                        tracker = updateTracker(colIdx, 'C', tracker, force=False)
                        modC = True
                        corrected_positions.append(colIdx)
                    add_nonrip[TinCol] += 1
                    _mark('non_rip_deamination', colIdx, 'T', TinCol, [0] * TinCol.size)

        if GAprop >= max_snp_noise:
            tg_rows = np.where(is_G & has_prev & (prev_base == bT))[0]
            _mark('rip_substrate', colIdx, 'G', tg_rows, prv[tg_rows] - colIdx)

            if GAprop > CTprop and is_G.any() and is_A.any():
                ta2_rows = np.where(is_A & has_prev & (prev_base == bT))[0]
                rev_TArows = ta2_rows
                AinCol = np.where(is_A)[0]

                if tg_rows.size and ta2_rows.size:
                    propRIPlike = (tg_rows.size + ta2_rows.size) / GA_count
                    add_rev[ta2_rows] += 1
                    _mark('rip_product', colIdx, 'A', ta2_rows, prv[ta2_rows] - colIdx)
                    nonrip_A = np.setdiff1d(AinCol, ta2_rows, assume_unique=True)
                    add_nonrip[nonrip_A] += 1
                    _mark(
                        'non_rip_deamination',
                        colIdx,
                        'A',
                        nonrip_A,
                        [0] * nonrip_A.size,
                    )
                    if propRIPlike >= min_rip_like or reaminate:
                        tracker = updateTracker(colIdx, 'G', tracker, force=False)
                        modG = True
                        corrected_positions.append(colIdx)
                else:
                    if reaminate:
                        tracker = updateTracker(colIdx, 'G', tracker, force=False)
                        modG = True
                        corrected_positions.append(colIdx)
                    add_nonrip[AinCol] += 1
                    _mark('non_rip_deamination', colIdx, 'A', AinCol, [0] * AinCol.size)

        if modC:
            maskedArr[np.where(is_T)[0] if reaminate else fwd_TArows, colIdx] = b'Y'
        if modG:
            maskedArr[np.where(is_A)[0] if reaminate else rev_TArows, colIdx] = b'R'

    nz = np.where((add_fwd != 0) | (add_rev != 0) | (add_nonrip != 0))[0]
    for r in nz.tolist():
        RIPcounts = updateRIPCount(
            r,
            RIPcounts,
            addRev=int(add_rev[r]),
            addFwd=int(add_fwd[r]),
            addNonRIP=int(add_nonrip[r]),
        )

    return (
        tracker,
        RIPcounts,
        _array_to_alignment(maskedArr, align),
        sorted(set(corrected_positions)),
        {cat: sorted(v) for cat, v in markup.items()},
    )


def random_alignment(rng, n_rows=None, n_cols=None):
    """Generate a random gapped nucleotide alignment biased toward RIP motifs."""
    n_rows = n_rows or int(rng.integers(2, 9))
    n_cols = n_cols or int(rng.integers(1, 26))
    # Weight A/C/T/G above gaps so RIP dinucleotide contexts occur often.
    alphabet = np.array(list('ACGTACGT-'))
    grid = rng.choice(alphabet, size=(n_rows, n_cols))
    return make_alignment([''.join(row) for row in grid])


def simulated_rip_alignment(rng, n_rows=None, n_cols=None):
    """
    Generate an alignment by applying simulated RIP to a common ancestor.

    Each row picks a strand and converts a random subset of that strand's
    substrate dinucleotides to TA, which is how real RIP variation arises. This
    populates the product branches of the scan far more densely than uniform
    random sequence does.
    """
    n_rows = n_rows or int(rng.integers(3, 10))
    n_cols = n_cols or int(rng.integers(20, 61))
    # Build the ancestor from dinucleotide blocks so RIP substrate is dense.
    blocks = rng.choice(['CA', 'TG', 'CG', 'AC', 'GT'], size=(n_cols + 1) // 2)
    ancestor = list(''.join(blocks))[:n_cols]

    rows = []
    for _ in range(n_rows):
        seq = list(ancestor)
        strand = rng.choice(['fwd', 'rev', 'none'])
        for i in range(len(seq) - 1):
            if strand == 'fwd' and seq[i] == 'C' and seq[i + 1] == 'A':
                if rng.random() < 0.6:
                    seq[i] = 'T'  # CA -> TA
            elif strand == 'rev' and seq[i] == 'T' and seq[i + 1] == 'G':
                if rng.random() < 0.6:
                    seq[i + 1] = 'A'  # TG -> TA
        # Sprinkle gaps and the occasional unrelated SNP, sparsely enough that
        # most dinucleotide contexts survive.
        for i in range(len(seq)):
            if rng.random() < 0.04:
                seq[i] = '-'
            elif rng.random() < 0.03:
                seq[i] = str(rng.choice(list('ACGT')))
        rows.append(''.join(seq))
    return make_alignment(rows)


# --------------------------------------------------------------------------
# Anti-drift: the vectorised classifier must reproduce the original scan
# --------------------------------------------------------------------------
PARAM_GRID = [
    (msn, mrl, ream)
    for msn in (0.3, 0.5, 0.7)
    for mrl in (0.0, 0.1, 0.5)
    for ream in (False, True)
]


def fuzz_corpus(rng, n=24):
    """Half uniform-random alignments, half simulated-RIP alignments."""
    return [
        random_alignment(rng) if i % 2 else simulated_rip_alignment(rng)
        for i in range(n)
    ]


def test_fuzz_corpus_exercises_every_markup_branch():
    """Guard: a green anti-drift test on an empty corpus would prove nothing."""
    rng = np.random.default_rng(20240709)
    totals = {'rip_product': 0, 'rip_substrate': 0, 'non_rip_deamination': 0}
    product_bases = set()
    corrected = fwd_cols = rev_cols = 0

    for align in fuzz_corpus(rng):
        cls = classify_alignment(align, progress=False)
        _, _, _, cp, markup = apply_classification(
            align, initTracker(align), initRIPCounter(align), cls
        )
        for cat in totals:
            totals[cat] += len(markup[cat])
        product_bases.update(p.base for p in markup['rip_product'])
        corrected += len(cp)
        fwd_cols += int(cls.fwd_col.sum())
        rev_cols += int(cls.rev_col.sum())

    assert totals['rip_product'] > 100, totals
    assert totals['rip_substrate'] > 400, totals
    assert totals['non_rip_deamination'] > 100, totals
    assert corrected > 50

    # Both strands must fire, or the anti-drift test only covers half the logic.
    assert product_bases == {'T', 'A'}, product_bases
    assert fwd_cols > 20 and rev_cols > 20


@pytest.mark.parametrize('max_snp_noise,min_rip_like,reaminate', PARAM_GRID)
def test_classifier_matches_reference_scan(max_snp_noise, min_rip_like, reaminate):
    """Vectorised classification reproduces the original per-column scan exactly."""
    rng = np.random.default_rng(20240709)

    for align in fuzz_corpus(rng):
        ref = _reference_correctRIP(
            align,
            initTracker(align),
            initRIPCounter(align),
            max_snp_noise,
            min_rip_like,
            reaminate,
        )
        cls = classify_alignment(
            align,
            max_snp_noise=max_snp_noise,
            min_rip_like=min_rip_like,
            reaminate=reaminate,
            progress=False,
        )
        got = apply_classification(
            align, initTracker(align), initRIPCounter(align), cls
        )

        ref_tracker, ref_counts, ref_masked, ref_corrected, ref_markup = ref
        got_tracker, got_counts, got_masked, got_corrected, got_markup = got

        assert [t.base for t in got_tracker.values()] == [
            t.base for t in ref_tracker.values()
        ]
        assert [str(r.seq) for r in got_masked] == [str(r.seq) for r in ref_masked]
        assert sorted(set(got_corrected)) == ref_corrected

        for cat in ref_markup:
            assert sorted(tuple(p) for p in got_markup[cat]) == ref_markup[cat], cat

        for row in ref_counts:
            assert got_counts[row].RIPcount == ref_counts[row].RIPcount
            assert got_counts[row].revRIPcount == ref_counts[row].revRIPcount
            assert got_counts[row].nonRIPcount == ref_counts[row].nonRIPcount


def test_markupdict_has_no_duplicate_cells():
    """Each (column, row) appears at most once per markup category."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        align = random_alignment(rng)
        cls = classify_alignment(align, progress=False)
        _, _, _, _, markup = apply_classification(
            align, initTracker(align), initRIPCounter(align), cls
        )
        for cat, positions in markup.items():
            cells = [(p.colIdx, p.rowIdx) for p in positions]
            assert len(cells) == len(set(cells)), cat


def test_markup_positions_sorted_by_column_then_row():
    """Markup lists are emitted in a deterministic column-major order."""
    align = make_alignment(['CACATG', 'TATATA', 'CATATG', 'TACATA'])
    cls = classify_alignment(align, progress=False)
    _, _, _, _, markup = apply_classification(
        align, initTracker(align), initRIPCounter(align), cls
    )
    for positions in markup.values():
        keys = [(p.colIdx, p.rowIdx) for p in positions]
        assert keys == sorted(keys)


# --------------------------------------------------------------------------
# Column blocking
# --------------------------------------------------------------------------
@pytest.mark.parametrize('block_size', [1, 2, 3, 7, 1000])
def test_block_size_is_bit_identical(block_size):
    """Column blocking never changes the result, only peak memory."""
    rng = np.random.default_rng(99)
    align = random_alignment(rng, n_rows=6, n_cols=23)
    arr = alignment_to_array(align)
    next_idx, prev_idx = _nongap_neighbors(arr)

    whole = classify_columns(arr, next_idx, prev_idx, progress=False)
    blocked = classify_columns(
        arr, next_idx, prev_idx, block_size=block_size, progress=False
    )

    for field in ('ca', 'ta', 'tg', 'ta2', 'fwd_col', 'rev_col', 'modC', 'modG'):
        assert np.array_equal(getattr(whole, field), getattr(blocked, field)), field


def test_blocking_spans_dinucleotides_across_block_boundary():
    """A dinucleotide straddling a block edge is still detected."""
    align = make_alignment(['CATG', 'TATA'])
    arr = alignment_to_array(align)
    next_idx, prev_idx = _nongap_neighbors(arr)
    # block_size=1 puts every base of every dinucleotide in a different block.
    blocked = classify_columns(arr, next_idx, prev_idx, block_size=1, progress=False)
    whole = classify_columns(arr, next_idx, prev_idx, progress=False)
    assert np.array_equal(blocked.ca, whole.ca)
    assert blocked.ca[0, 0]  # seq0 col0: C followed by A
    assert blocked.tg[0, 3]  # seq0 col3: G preceded by T


# --------------------------------------------------------------------------
# Strand exclusivity: the strict `>` is load-bearing
# --------------------------------------------------------------------------
def test_forward_and_reverse_correction_are_mutually_exclusive():
    """A column is never corrected on both strands, so Y/R masks cannot collide."""
    rng = np.random.default_rng(11)
    for reaminate in (False, True):
        for _ in range(40):
            align = random_alignment(rng)
            cls = classify_alignment(align, reaminate=reaminate, progress=False)
            assert not (cls.modC & cls.modG).any()
            assert not (cls.fwd_col & cls.rev_col).any()
            assert not (cls.mask_Y & cls.mask_R).any()


def test_fifty_fifty_column_is_corrected_on_neither_strand():
    """With C/T and G/A exactly balanced, neither strand claims a strict majority."""
    # Column 0: two C, two G -> CTprop == GAprop == 0.5
    align = make_alignment(['CA', 'CA', 'GT', 'GT'])
    cls = classify_alignment(align, max_snp_noise=0.5, progress=False)
    assert not cls.modC[0]
    assert not cls.modG[0]
    # Both gates pass at exactly 0.5, so substrate is still observable.
    assert cls.ct_ok[0] and cls.ga_ok[0]


# --------------------------------------------------------------------------
# Dinucleotide semantics: gaps and CpG
# --------------------------------------------------------------------------
def test_dinucleotide_skips_gap_columns():
    """C-A spanning a gap column is a CpA substrate; neighbours are non-gap."""
    align = make_alignment(['C--A', 'T--A'])
    cls = classify_alignment(align, progress=False)
    assert cls.ca[0, 0], 'C at col0 followed by non-gap A is a substrate'
    assert cls.ta[1, 0], 'T at col0 followed by non-gap A is a product candidate'
    assert cls.fwd_col[0]


def test_gap_disrupted_rows_are_scored_per_row():
    """Rows differ in where their non-gap neighbour lies; each is scored on its own."""
    align = make_alignment(['CA-', 'C-A', 'TA-'])
    cls = classify_alignment(align, progress=False)
    # Every row has C or T at col0 with a non-gap A downstream.
    assert cls.ca[0, 0] and cls.ca[1, 0] and cls.ta[2, 0]


def test_cpg_is_neither_substrate_nor_product():
    """CG is neither: its C is not followed by A, its G not preceded by T."""
    align = make_alignment(['CG', 'CG'])
    cls = classify_alignment(align, progress=False)
    assert not cls.ca.any()
    assert not cls.tg.any()
    assert not cls.ta.any()
    assert not cls.ta2.any()


def test_cpg_diverged_forms_are_both_substrate():
    """A CG that diverged to CA or TG yields substrate on the respective strand."""
    align = make_alignment(['CA', 'TG', 'CG'])
    cls = classify_alignment(align, max_snp_noise=0.4, progress=False)
    assert cls.ca[0, 0], 'CG -> CA gives a forward substrate'
    assert cls.tg[1, 1], 'CG -> TG gives a reverse substrate'
    # The ancestral CG row contributes to neither.
    assert not cls.ca[2, 0] and not cls.tg[2, 1]


def test_fully_converted_column_is_not_a_rip_column():
    """With no surviving CA anywhere, the alignment holds no evidence of RIP."""
    align = make_alignment(['TA', 'TA', 'TA'])
    cls = classify_alignment(align, progress=False)
    assert not cls.fwd_col.any()
    assert not cls.prod_fwd.any()


# --------------------------------------------------------------------------
# RSI algebra
# --------------------------------------------------------------------------
def test_rsi_zero_when_no_rip():
    """Unmutated substrate on both strands scores 0, not NaN."""
    cls = classify_alignment(make_alignment(['CACTG'] * 3), progress=False)
    res = compute_rsi(cls)
    assert np.allclose(res.p_fwd, 0.0)
    assert np.allclose(res.p_rev, 0.0)
    assert np.allclose(res.rsi, 0.0)


def test_rsi_plus_one_for_pure_forward_rip():
    """A sequence with every CA converted and every TG intact scores +1."""
    cls = classify_alignment(
        make_alignment(['CAGTG', 'TAGTG', 'TAGTG']), progress=False
    )
    res = compute_rsi(cls)
    # Row 0 retains its substrate; rows 1 and 2 are fully forward-RIP'd.
    assert res.rsi[0] == pytest.approx(0.0)
    assert res.rsi[1] == pytest.approx(1.0)
    assert res.rsi[2] == pytest.approx(1.0)


def test_rsi_minus_one_for_pure_reverse_rip():
    """A sequence with every TG converted and every CA intact scores -1."""
    cls = classify_alignment(
        make_alignment(['CATTG', 'CATTA', 'CATTA']), progress=False
    )
    res = compute_rsi(cls)
    assert res.rsi[0] == pytest.approx(0.0)
    assert res.rsi[1] == pytest.approx(-1.0)
    assert res.rsi[2] == pytest.approx(-1.0)


def test_rsi_zero_when_both_strands_fully_converted():
    """Full conversion on both strands is neutral, like no conversion at all."""
    # Row 3 lost both its CA (col0) and its TG (col4); rows 0-2 retain evidence.
    cls = classify_alignment(
        make_alignment(['CACTG', 'CACTG', 'CACTG', 'TACTA']), progress=False
    )
    res = compute_rsi(cls, ambiguous='split')
    assert res.rsi[3] == pytest.approx(0.0)
    # ...but the components separate the two cases.
    assert res.p_fwd[3] == pytest.approx(1.0)
    assert res.p_rev[3] == pytest.approx(1.0)
    assert res.p_fwd[0] == pytest.approx(0.0)
    assert res.p_rev[0] == pytest.approx(0.0)


def test_rsi_nan_when_a_strand_has_no_evidence():
    """No substrate and no product on a strand yields NaN, not a spurious 0."""
    # Row 0 has a CA but nothing resembling a reverse site.
    cls = classify_alignment(make_alignment(['CAA', 'CAA']), progress=False)
    res = compute_rsi(cls)
    assert np.all(np.isnan(res.p_rev))
    assert np.all(np.isnan(res.rsi))
    assert np.allclose(res.p_fwd, 0.0)


def test_rsi_bounded_by_plus_minus_one():
    """RSI never leaves [-1, 1] under any policy."""
    rng = np.random.default_rng(4242)
    for _ in range(30):
        cls = classify_alignment(random_alignment(rng), progress=False)
        for policy in AMBIGUITY_POLICIES:
            rsi = compute_rsi(cls, ambiguous=policy).rsi
            finite = rsi[~np.isnan(rsi)]
            assert np.all(finite >= -1.0) and np.all(finite <= 1.0), policy


# --------------------------------------------------------------------------
# Ambiguity policies
# --------------------------------------------------------------------------
SPEC_EXAMPLE = ['CA', 'TA', 'TA', 'TA', 'TG', 'CG']


def test_spec_example_is_strand_symmetric_under_every_policy():
    """The worked example from the design spec: RSI = 0 whichever policy is used."""
    cls = classify_alignment(make_alignment(SPEC_EXAMPLE), progress=False)

    assert cls.fwd_col[0] and cls.rev_col[1]
    assert not cls.fwd_col[1] and not cls.rev_col[0]

    expected_p = {'split': 0.60, 'exclude': 0.00, 'weight': 0.60, 'both': 0.75}
    for policy, p in expected_p.items():
        pooled = compute_rsi(cls, ambiguous=policy).pooled()
        assert pooled['n_ambiguous'] == 3
        assert pooled['p_fwd'] == pytest.approx(p)
        assert pooled['p_rev'] == pytest.approx(p)
        assert pooled['RSI'] == pytest.approx(0.0)


# An asymmetric alignment where the four policies give four different answers.
# Columns 0 and 2 are forward RIP columns; column 1 is a reverse RIP column.
# The two TA dinucleotides at (col0, col1) are ambiguous, and the evidence is
# lopsided: three unmutated C at col0 against a single unmutated G at col1.
ASYMMETRIC_EXAMPLE = ['CACA', 'CATA', 'CATA', 'TATA', 'TATA', 'TGCG']

ASYMMETRIC_EXPECTED = {
    #            fwd_prod  rev_prod  p_fwd    p_rev    RSI
    'split': (5.0, 1.0, 0.555556, 0.500000, +0.055556),
    'exclude': (4.0, 0.0, 0.500000, 0.000000, +0.500000),
    'weight': (5.5, 0.5, 0.578947, 0.333333, +0.245614),
    'both': (6.0, 2.0, 0.600000, 0.666667, -0.066667),
}


@pytest.mark.parametrize('policy', AMBIGUITY_POLICIES)
def test_ambiguity_policies_give_distinct_pinned_values(policy):
    """Each policy resolves the same ambiguous TAs to a different imbalance."""
    cls = classify_alignment(make_alignment(ASYMMETRIC_EXAMPLE), progress=False)
    pooled = compute_rsi(cls, ambiguous=policy).pooled()

    fwd_prod, rev_prod, p_fwd, p_rev, rsi = ASYMMETRIC_EXPECTED[policy]
    assert pooled['n_ambiguous'] == 2
    assert pooled['fwd_product'] == pytest.approx(fwd_prod)
    assert pooled['rev_product'] == pytest.approx(rev_prod)
    assert pooled['fwd_substrate'] == pytest.approx(4.0)
    assert pooled['rev_substrate'] == pytest.approx(1.0)
    assert pooled['p_fwd'] == pytest.approx(p_fwd, abs=1e-6)
    assert pooled['p_rev'] == pytest.approx(p_rev, abs=1e-6)
    assert pooled['RSI'] == pytest.approx(rsi, abs=1e-6)


def test_policy_choice_can_flip_the_sign_of_rsi():
    """The four policies are not interchangeable: 'exclude' and 'both' disagree on sign."""
    cls = classify_alignment(make_alignment(ASYMMETRIC_EXAMPLE), progress=False)
    assert compute_rsi(cls, ambiguous='exclude').pooled()['RSI'] > 0
    assert compute_rsi(cls, ambiguous='both').pooled()['RSI'] < 0


def test_weight_policy_follows_the_evidence():
    """'weight' assigns an ambiguous TA in proportion to surviving substrate."""
    cls = classify_alignment(make_alignment(ASYMMETRIC_EXAMPLE), progress=False)
    # col0 holds three unmutated C, col1 a single unmutated G -> w_fwd = 3/4.
    assert int(cls.nC[0]) == 3
    assert int(cls.nG[1]) == 1

    excl = compute_rsi(cls, ambiguous='exclude').pooled()
    wtd = compute_rsi(cls, ambiguous='weight').pooled()
    n_amb = wtd['n_ambiguous']
    assert wtd['fwd_product'] - excl['fwd_product'] == pytest.approx(0.75 * n_amb)
    assert wtd['rev_product'] - excl['rev_product'] == pytest.approx(0.25 * n_amb)


def test_split_and_both_agree_on_symmetric_ambiguity():
    """When ambiguity is strand-symmetric, split and both give the same RSI."""
    cls = classify_alignment(make_alignment(SPEC_EXAMPLE), progress=False)
    a = compute_rsi(cls, ambiguous='split').pooled()['RSI']
    b = compute_rsi(cls, ambiguous='both').pooled()['RSI']
    assert a == pytest.approx(b)


def test_exclude_drops_ambiguous_from_numerator_only():
    """'exclude' removes ambiguous products but keeps directly observed substrate."""
    cls = classify_alignment(make_alignment(SPEC_EXAMPLE), progress=False)
    pooled = compute_rsi(cls, ambiguous='exclude').pooled()
    assert pooled['fwd_product'] == 0.0
    assert pooled['fwd_substrate'] == 1.0
    assert pooled['rev_substrate'] == 1.0


def test_invalid_policy_raises():
    """Unknown options are rejected rather than silently defaulting."""
    cls = classify_alignment(make_alignment(SPEC_EXAMPLE), progress=False)
    with pytest.raises(ValueError, match='ambiguous must be one of'):
        compute_rsi(cls, ambiguous='nonsense')
    with pytest.raises(ValueError, match='substrate_scope must be one of'):
        compute_rsi(cls, substrate_scope='nonsense')


# --------------------------------------------------------------------------
# Substrate scope
# --------------------------------------------------------------------------
@pytest.mark.parametrize('scope', SUBSTRATE_SCOPES)
def test_substrate_scope_options_all_run(scope):
    """Every documented substrate scope produces a finite, bounded result."""
    cls = classify_alignment(make_alignment(ASYMMETRIC_EXAMPLE), progress=False)
    res = compute_rsi(cls, substrate_scope=scope)
    finite = res.rsi[~np.isnan(res.rsi)]
    assert np.all(np.abs(finite) <= 1.0)


def test_rip_like_columns_scope_yields_nan_for_unripped_sequence():
    """The narrow scope cannot score a sequence that has no RIP at all."""
    cls = classify_alignment(make_alignment(['CACTG'] * 3), progress=False)

    # Substrates are directly observed, so the default scope scores this as 0.
    assert np.allclose(compute_rsi(cls, substrate_scope='all').rsi, 0.0)

    # Restricting substrates to RIP columns leaves nothing in the denominator.
    narrow = compute_rsi(cls, substrate_scope='rip_like_columns')
    assert np.all(np.isnan(narrow.rsi))


def test_substrate_scope_all_counts_substrate_outside_rip_columns():
    """A CA in a column with no aligned product still counts as substrate."""
    align = make_alignment(['CAAA', 'CAAA'])
    cls = classify_alignment(align, progress=False)
    assert not cls.fwd_col.any()
    res = compute_rsi(cls, substrate_scope='all')
    assert np.allclose(res.fwd_sub, 1.0)
    assert np.allclose(res.p_fwd, 0.0)


# --------------------------------------------------------------------------
# Significance
# --------------------------------------------------------------------------
def test_pvalue_is_one_when_proportions_are_identical():
    """No asymmetry means no evidence against the null."""
    cls = classify_alignment(make_alignment(SPEC_EXAMPLE), progress=False)
    pooled = compute_rsi(cls).pooled()
    assert pooled['z'] == pytest.approx(0.0)
    assert pooled['pvalue'] == pytest.approx(1.0)


def test_pvalue_is_one_when_every_site_converted_on_both_strands():
    """A degenerate pooled proportion of 1 gives zero standard error, p = 1."""
    cls = classify_alignment(
        make_alignment(['CACTG', 'CACTG', 'CACTG', 'TACTA']), progress=False
    )
    res = compute_rsi(cls)
    assert res.z[3] == pytest.approx(0.0)
    assert res.pvalue[3] == pytest.approx(1.0)


def test_pvalue_nan_when_a_strand_has_no_trials():
    """An undefined proportion cannot be tested."""
    cls = classify_alignment(make_alignment(['CAA', 'CAA']), progress=False)
    res = compute_rsi(cls)
    assert np.all(np.isnan(res.pvalue))
    assert np.all(np.isnan(res.z))


def test_strong_forward_bias_is_significant():
    """Many forward conversions against intact reverse sites is detectable."""
    # 20 forward CA sites converted in row 1; 20 reverse TG sites untouched.
    fwd_sites = 'CA' * 20
    rev_sites = 'TG' * 20
    cls = classify_alignment(
        make_alignment([fwd_sites + rev_sites, 'TA' * 20 + rev_sites]), progress=False
    )
    res = compute_rsi(cls)
    assert res.rsi[1] == pytest.approx(1.0)
    assert res.pvalue[1] < 0.01


def test_as_records_round_trips_ids():
    """as_records() labels each row with its sequence id."""
    align = make_alignment(SPEC_EXAMPLE)
    cls = classify_alignment(align, progress=False)
    records = compute_rsi(cls).as_records([r.id for r in align])
    assert [r['id'] for r in records] == [f'seq{i}' for i in range(6)]
    assert records[1]['n_ambiguous'] == 1
