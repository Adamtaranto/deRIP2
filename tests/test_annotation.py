"""
Tests for GFF3 parsing, gap-coordinate mapping, and RIP effect prediction.

Effect-type correctness is exercised with hand-built rows and genes, where the
ancestral and observed coding sequences are fully controlled, so each effect
kind (missense, premature stop, frameshift, splice-site) is provoked in
isolation and on both strands. Parsing and gap mapping are checked against the
committed ``mintest.gff3`` fixture.
"""

import logging

import numpy as np
import pytest

from derip2.annotation import (
    Feature,
    Gene,
    parse_gff3,
    predict_gene_effects,
    translate_cds,
    ungapped_to_column_map,
    warn_unmatched_seqids,
)

logging.disable(logging.CRITICAL)


def row(seq):
    """Build an 'S1' byte-array alignment row from a string."""
    return np.frombuffer(seq.encode('ascii'), dtype='S1').copy()


def cds_gene(gene_id, seqid, strand, exons, phase=0):
    """Build a single-transcript Gene from ``(start, end)`` exon tuples."""
    features = [
        Feature(seqid, 'CDS', s, e, strand, phase, {}, f'{gene_id}:{i}', gene_id)
        for i, (s, e) in enumerate(exons)
    ]
    return Gene(gene_id=gene_id, seqid=seqid, strand=strand, cds=features)


# --- parsing ---------------------------------------------------------------


def test_parse_gff3_structure(gff_path):
    """The fixture parses into the expected genes, grouped by sequence."""
    genes = parse_gff3(gff_path)
    assert set(genes) == {'Seq1', 'Seq3', 'Seq4'}
    assert [g.gene_id for g in genes['Seq1']] == ['mRNA1']
    # gene3 is multi-exon on the plus strand.
    gene3 = genes['Seq4'][0]
    assert gene3.strand == '+'
    assert len(gene3.cds) == 2
    assert [(c.start, c.end) for c in gene3.cds] == [(1, 6), (13, 21)]
    # gene2 is on the minus strand.
    assert genes['Seq3'][0].strand == '-'


def test_parse_gff3_attribute_unescape(tmp_path):
    """Percent-encoded attribute values are unescaped."""
    gff = tmp_path / 'x.gff3'
    gff.write_text(
        '##gff-version 3\n'
        'chr1\ttest\tCDS\t1\t9\t.\t+\t0\tID=c1;Parent=g1;Note=hello%20world\n'
    )
    genes = parse_gff3(str(gff))
    assert genes['chr1'][0].cds[0].attributes['Note'] == 'hello world'


def test_parse_gff3_bad_coordinate(tmp_path):
    """A non-integer coordinate is a hard error."""
    gff = tmp_path / 'bad.gff3'
    gff.write_text('chr1\tt\tCDS\tx\t9\t.\t+\t0\tID=c\n')
    with pytest.raises(ValueError):
        parse_gff3(str(gff))


def test_warn_unmatched_seqids(gff_path):
    """Sequence ids absent from the alignment are reported."""
    genes = parse_gff3(gff_path)
    unmatched = warn_unmatched_seqids(genes, ['Seq1', 'Seq3'])
    assert unmatched == ['Seq4']


# --- gap-coordinate mapping ------------------------------------------------


def test_ungapped_to_column_map():
    """Ungapped positions map onto the correct gapped columns."""
    r = row('A-CG--T')
    m = ungapped_to_column_map(r)
    # bases at columns 0,2,3,6.
    assert m.tolist() == [0, 2, 3, 6]
    # 1-based ungapped position 3 (the G) -> column m[2] == 3.
    assert int(m[3 - 1]) == 3


# --- effect prediction -----------------------------------------------------


def test_premature_stop_plus():
    """A CAA->TAA change on the plus strand is a premature stop."""
    ref = row('ATGCAAGGG')  # M Q G
    tgt = row('ATGTAAGGG')  # M * G
    gene = cds_gene('g', 'S', '+', [(1, 9)])
    effects = predict_gene_effects(
        gene, tgt, ref, ungapped_to_column_map(tgt), seq_id='S'
    )
    assert len(effects) == 1
    assert effects[0].kind == 'premature_stop'
    assert effects[0].aa_pos == 2
    assert effects[0].ref_aa == 'Q' and effects[0].alt_aa == '*'


def test_missense_plus():
    """A CAA->CAT change is a missense (Q->H)."""
    ref = row('ATGCAA')  # M Q
    tgt = row('ATGCAT')  # M H
    gene = cds_gene('g', 'S', '+', [(1, 6)])
    effects = predict_gene_effects(
        gene, tgt, ref, ungapped_to_column_map(tgt), seq_id='S'
    )
    assert [e.kind for e in effects] == ['missense']
    assert effects[0].ref_aa == 'Q' and effects[0].alt_aa == 'H'


def test_minus_strand_premature_stop():
    """A minus-strand gene is read as the reverse complement."""
    ref = row('TTGCAT')  # revcomp -> ATGCAA -> M Q
    tgt = row('TTACAT')  # revcomp -> ATGTAA -> M *
    gene = cds_gene('g', 'S', '-', [(1, 6)])
    effects = predict_gene_effects(
        gene, tgt, ref, ungapped_to_column_map(tgt), seq_id='S'
    )
    assert [e.kind for e in effects] == ['premature_stop']


def test_frameshift_from_gap():
    """A gap that changes the coding length by a non-triplet is a frameshift."""
    ref = row('ATGAAACCCGGG')  # 12 nt
    tgt = row('ATG-AACCCGGG')  # ungapped 11 nt -> frame shifted
    gene = cds_gene('g', 'S', '+', [(1, 11)])
    effects = predict_gene_effects(
        gene, tgt, ref, ungapped_to_column_map(tgt), seq_id='S'
    )
    assert any(e.kind == 'frameshift' for e in effects)


def test_splice_site_broken_donor():
    """Breaking a canonical GT donor is reported as a splice-site effect."""
    ref = row('ATGGTCCAGAAA')  # exon1 ATG | intron GT..AG | exon2 AAA
    tgt = row('ATGATCCAGAAA')  # donor GT -> AT
    gene = cds_gene('g', 'S', '+', [(1, 3), (10, 12)])
    effects = predict_gene_effects(
        gene, tgt, ref, ungapped_to_column_map(tgt), seq_id='S'
    )
    splice = [e for e in effects if e.kind == 'splice_site']
    assert len(splice) == 1
    assert splice[0].nt_ref == 'GT' and splice[0].nt_alt == 'AT'


def test_no_effect_when_identical():
    """Identical target and reference yield no effects."""
    seq = row('ATGCAAGGG')
    gene = cds_gene('g', 'S', '+', [(1, 9)])
    effects = predict_gene_effects(
        gene, seq, seq, ungapped_to_column_map(seq), seq_id='S'
    )
    assert effects == []


def test_synonymous_optional():
    """Synonymous changes are only reported when requested."""
    ref = row('ATGTTA')  # M L (TTA)
    tgt = row('ATGCTA')  # M L (CTA) -> synonymous
    gene = cds_gene('g', 'S', '+', [(1, 6)])
    m = ungapped_to_column_map(tgt)
    assert predict_gene_effects(gene, tgt, ref, m, seq_id='S') == []
    syn = predict_gene_effects(gene, tgt, ref, m, seq_id='S', include_synonymous=True)
    assert [e.kind for e in syn] == ['synonymous']


def test_gene_off_sequence_is_skipped():
    """A gene whose exon runs past the sequence is skipped, not crashed."""
    ref = row('ATGCAA')
    tgt = row('ATGCAA')
    gene = cds_gene('g', 'S', '+', [(1, 99)])  # far past the 6-nt sequence
    assert predict_gene_effects(gene, tgt, ref, ungapped_to_column_map(tgt)) == []
    assert translate_cds(gene, ref, ungapped_to_column_map(ref)) == ''


# --- translation -----------------------------------------------------------


def test_translate_cds_standard_code():
    """CDS translation renders stop codons as '*'."""
    seq = row('ATGAAATAG')  # M K *
    gene = cds_gene('g', 'S', '+', [(1, 9)])
    assert translate_cds(gene, seq, ungapped_to_column_map(seq)) == 'MK*'


def test_translate_cds_alternate_code():
    """A non-standard genetic code changes the translation."""
    # TGA is a stop under table 1 but Trp (W) under table 4 (mould mito).
    seq = row('ATGTGA')
    gene = cds_gene('g', 'S', '+', [(1, 6)])
    m = ungapped_to_column_map(seq)
    assert translate_cds(gene, seq, m, genetic_code=1) == 'M*'
    assert translate_cds(gene, seq, m, genetic_code=4) == 'MW'


def test_translate_multi_exon_joins_exons():
    """Exons are concatenated in transcription order before translation."""
    seq = row('ATGGTCCAGAAA')
    gene = cds_gene('g', 'S', '+', [(1, 3), (10, 12)])
    # ATG + AAA -> M K
    assert translate_cds(gene, seq, ungapped_to_column_map(seq)) == 'MK'


def test_phase_trims_leading_bases():
    """A non-zero first-exon phase trims leading bases before translation."""
    seq = row('GGATGAAATAG')  # first 2 bases are UTR-ish; phase 2 skips them
    gene = cds_gene('g', 'S', '+', [(1, 11)], phase=2)
    # After trimming 2 bases: ATGAAATAG -> M K *
    assert translate_cds(gene, seq, ungapped_to_column_map(seq)) == 'MK*'
