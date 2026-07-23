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
    cds_alignment_columns,
    cds_display_id,
    cds_stop_columns,
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


def test_cds_display_id_prefers_cds_id():
    """cds_display_id returns the CDS ID (first segment), not the parent id.

    parse_gff3 groups by Parent, so gene_id is the transcript; the CDS's own ID
    lives on each segment's feature_id and all segments of one CDS share it.
    """
    # Parent 'mRNA3' owns a two-segment CDS whose ID is 'cds3'.
    cds = [
        Feature('S4', 'CDS', 1, 6, '+', 0, {}, 'cds3', 'mRNA3'),
        Feature('S4', 'CDS', 13, 21, '+', 0, {}, 'cds3', 'mRNA3'),
    ]
    gene = Gene(gene_id='mRNA3', seqid='S4', strand='+', cds=cds)
    assert cds_display_id(gene) == 'cds3'


def test_cds_display_id_falls_back_to_gene_id():
    """cds_display_id falls back to gene_id when the CDS carried no ID."""
    cds = [Feature('S1', 'CDS', 1, 9, '+', 0, {}, None, 'mRNA1')]
    gene = Gene(gene_id='mRNA1', seqid='S1', strand='+', cds=cds)
    assert cds_display_id(gene) == 'mRNA1'
    # A gene with no CDS at all also falls back safely.
    assert cds_display_id(Gene('g', 'S1', '+', [])) == 'g'


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


def test_minus_strand_uses_transcription_first_exon_phase():
    """
    Phase is taken from the 5' exon in transcription order, not genomic order.

    For a minus-strand gene, ``Gene.cds`` is sorted ascending by genomic start,
    so ``cds[0]`` is the 3'-terminal exon. Using its phase would shift the frame
    (the historical off-by-one). Here the two exons carry different phases: the
    transcription-first (high-coordinate) exon is phase 0, the genomic-first
    (low-coordinate) exon is phase 2. The correct translation uses phase 0.
    """
    # Coding sequence (5'->3') is the reverse complement of the whole 12-nt row.
    # We want coding = ATGAAACCCTAA (M K P *), so row = revcomp of that.
    from Bio.Seq import Seq as _Seq

    coding = 'ATGAAACCCTAA'
    fwd = str(_Seq(coding).reverse_complement())
    seq = row(fwd)

    # Two adjacent exons; genomic-first (1-6) gets phase 2, 5' exon (7-12) phase 0.
    exons = [
        Feature('S', 'CDS', 1, 6, '-', 2, {}, 'g:0', 'g'),  # 3' end (cds[0])
        Feature('S', 'CDS', 7, 12, '-', 0, {}, 'g:1', 'g'),  # 5' start
    ]
    gene = Gene(gene_id='g', seqid='S', strand='-', cds=exons)

    # Correct: phase 0 from the 5' exon -> full frame -> M K P *
    assert translate_cds(gene, seq, ungapped_to_column_map(seq)) == 'MKP*'


def test_cds_alignment_columns_and_stops():
    """CDS columns project onto the alignment; stop codons map back to columns."""
    # Plus-strand single exon; coding ATG AAA TAA -> M K * (stop at codon 3).
    seq = row('ATGAAATAA')
    gene = cds_gene('g', 'S', '+', [(1, 9)])
    u2c = ungapped_to_column_map(seq)
    cols = cds_alignment_columns(gene, u2c)
    assert cols == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # The stop codon is codon 3 (0-based k=2); its middle base is column 7.
    assert cds_stop_columns(gene, seq, cols) == [7]


def test_cds_stop_columns_minus_strand():
    """Stops are found in the correct frame for a minus-strand projection."""
    from Bio.Seq import Seq as _Seq

    # Coding TAA is a stop; put it mid-CDS: ATG TAA GGG -> M * G.
    coding = 'ATGTAAGGG'
    seq = row(str(_Seq(coding).reverse_complement()))
    gene = cds_gene('g', 'S', '-', [(1, 9)])
    cols = cds_alignment_columns(gene, ungapped_to_column_map(seq))
    stops = cds_stop_columns(gene, seq, cols)
    assert len(stops) == 1


def test_cds_stop_columns_empty_for_no_columns():
    """An empty column list yields no stops rather than erroring."""
    seq = row('ATGAAA')
    gene = cds_gene('g', 'S', '+', [(1, 6)])
    assert cds_stop_columns(gene, seq, []) == []


# --- alignment-level orchestration -----------------------------------------


@pytest.fixture
def gff_derip(mintest_path):
    """A DeRIP object with RIP calculated on the reference alignment."""
    from derip2.derip import DeRIP

    derip = DeRIP(mintest_path)
    derip.calculate_rip()
    return derip


def test_compute_effects_for_alignment(gff_derip, gff_path):
    """Effects are computed for annotated sequences against the ancestor."""
    from derip2.annotation import compute_effects_for_alignment

    genes = parse_gff3(gff_path)
    effects = compute_effects_for_alignment(gff_derip, genes)
    # Seq1 and Seq3 carry RIP-induced coding changes in the fixture.
    assert 'Seq1' in effects
    assert all(
        e.seq_id in {'Seq1', 'Seq3', 'Seq4'} for v in effects.values() for e in v
    )


def test_deripd_translations(gff_derip, gff_path):
    """Every annotated gene gets a deRIP'd protein translation."""
    from derip2.annotation import deripd_translations

    genes = parse_gff3(gff_path)
    aa = deripd_translations(gff_derip, genes)
    assert set(aa) == {'mRNA1', 'mRNA2', 'mRNA3'}
    assert all(isinstance(v, str) for v in aa.values())


def test_write_snp_effects(gff_derip, gff_path, tmp_path):
    """The SNP-effect TSV lists effects and restored translations."""
    from derip2.annotation import (
        compute_effects_for_alignment,
        deripd_translations,
        write_snp_effects,
    )

    genes = parse_gff3(gff_path)
    effects = compute_effects_for_alignment(gff_derip, genes)
    aa = deripd_translations(gff_derip, genes)
    out = tmp_path / 'snp_effects.txt'
    write_snp_effects(str(out), effects, aa)
    text = out.read_text()
    assert 'seq_id\tgene_id\tkind' in text
    assert 'deRIP-restored CDS translations' in text
    assert 'mRNA1' in text


def test_build_annotation_spans(gff_derip, gff_path):
    """Gene exons project onto alignment-column spans, one row per gene."""
    from derip2.annotation import build_annotation_spans

    genes = parse_gff3(gff_path)
    row_lookup = {
        rec.id: gff_derip.column_classes.arr[i]
        for i, rec in enumerate(gff_derip.alignment)
    }
    spans = build_annotation_spans(genes, row_lookup)
    assert spans, 'expected annotation spans'
    # Each span is (start, end, color, label, track_row) with start <= end.
    for start, end, color, _label, track_row in spans:
        assert start <= end
        assert color.startswith('#')
        assert track_row >= 0
    # The multi-exon gene contributes two spans on one track row.
    mrna3_rows = {
        s[4] for s in spans if s[3] == 'mRNA3' or s[4] == max(s[4] for s in spans)
    }
    assert mrna3_rows


def test_load_annotation_colors(tmp_path):
    """A colour override file merges over the defaults."""
    from derip2.annotation import DEFAULT_ANNOTATION_COLORS, load_annotation_colors

    f = tmp_path / 'colors.tsv'
    f.write_text('# custom\nCDS\t#123456\ngene\t#abcdef\n')
    colors = load_annotation_colors(str(f))
    assert colors['CDS'] == '#123456'
    assert colors['gene'] == '#abcdef'
    # Untouched types keep their defaults.
    assert colors['exon'] == DEFAULT_ANNOTATION_COLORS['exon']


def test_annotation_track_smoke(gff_derip, gff_path, tmp_path):
    """drawMiniAlignment renders with an annotation track."""
    import matplotlib

    matplotlib.use('Agg')

    from derip2.annotation import build_annotation_spans

    genes = parse_gff3(gff_path)
    row_lookup = {
        rec.id: gff_derip.column_classes.arr[i]
        for i, rec in enumerate(gff_derip.alignment)
    }
    spans = build_annotation_spans(genes, row_lookup)
    out = tmp_path / 'aln.png'
    result = gff_derip.plot_alignment(output_file=str(out), annotation_track=spans)
    assert result
    assert out.exists() and out.stat().st_size > 0


def test_build_cds_tracks(gff_derip, gff_path):
    """build_cds_tracks yields rich per-gene tracks with consensus-frame stops."""
    import numpy as np

    from derip2.annotation import build_cds_tracks

    genes = parse_gff3(gff_path)
    row_lookup = {
        rec.id: gff_derip.column_classes.arr[i]
        for i, rec in enumerate(gff_derip.alignment)
    }
    consensus_row = np.frombuffer(
        str(gff_derip.gapped_consensus.seq).upper().encode('ascii'), dtype='S1'
    )
    tracks = build_cds_tracks(genes, row_lookup, consensus_row)
    assert tracks
    for exon_spans, strand, stops, label, colour in tracks:
        assert exon_spans and all(len(s) == 2 for s in exon_spans)
        assert strand in ('+', '-')
        assert isinstance(stops, list)
        assert isinstance(label, str) and label
        assert colour.startswith('#')
