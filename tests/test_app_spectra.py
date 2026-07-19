"""
CLI tests for the ``derip2-spectra`` entry point.

These drive the command through Click's ``CliRunner`` and check that the matrix
files, tables and figures are produced and are well-formed.
"""

import logging
import os
import shutil

from click.testing import CliRunner
import pytest

from derip2.app_spectra import main
from derip2.spectra import read_sbs_matrix

logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
MINTEST = os.path.join(HERE, 'data', 'mintest.fa')

_HAVE_IQTREE = any(shutil.which(name) for name in ('iqtree3', 'iqtree2', 'iqtree'))


def test_cli_produces_all_outputs(tmp_path):
    """A default run writes both matrices, both tables and the figures."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ['-i', MINTEST, '-d', str(tmp_path), '-p', 'mt', '--sbs', 'both'],
    )
    assert result.exit_code == 0, result.output
    produced = set(os.listdir(tmp_path))
    for name in (
        'mt.SBS96.txt',
        'mt.SBS192.txt',
        'mt_events.tsv',
        'mt_homoplasy.tsv',
        'mt_SBS96.png',
        'mt_SBS192.png',
        'mt_strand_asymmetry.png',
        'mt_homoplasy.png',
    ):
        assert name in produced, f'missing {name}'

    channels, samples, matrix = read_sbs_matrix(str(tmp_path / 'mt.SBS96.txt'))
    assert len(channels) == 96
    assert samples == ['AllSequences']
    assert matrix.shape == (96, 1)


def test_cli_downstream_context_outputs(tmp_path):
    """--context downstream writes the downstream matrix, sidecar and plot only."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ['-i', MINTEST, '-d', str(tmp_path), '-p', 'ds', '--context', 'downstream'],
    )
    assert result.exit_code == 0, result.output
    produced = set(os.listdir(tmp_path))
    for name in (
        'ds.SBSdownstream.txt',
        'ds.SBSdownstream.meta.json',
        'ds_events.tsv',
        'ds_homoplasy.tsv',
        'ds_SBSdownstream.png',
        'ds_homoplasy.png',
    ):
        assert name in produced, f'missing {name}'
    # No SBS-96/192 or strand-asymmetry outputs in the downstream context.
    assert not any(n.startswith('ds.SBS96') for n in produced)
    assert not any(n.startswith('ds.SBS192') for n in produced)
    assert 'ds_strand_asymmetry.png' not in produced

    channels, samples, matrix = read_sbs_matrix(str(tmp_path / 'ds.SBSdownstream.txt'))
    assert len(channels) == 96
    assert channels[0].startswith('[')  # distinct downstream label form
    assert matrix.shape == (96, 1)

    import json

    with open(tmp_path / 'ds.SBSdownstream.meta.json') as handle:
        meta = json.load(handle)
    assert meta['context'] == 'downstream'
    assert meta['kind'] == 'downstream'


def test_cli_downstream_rejects_explicit_sbs192(tmp_path):
    """--context downstream with an explicit --sbs 192/both is a clear error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            MINTEST,
            '-d',
            str(tmp_path),
            '-p',
            'ds',
            '--context',
            'downstream',
            '--sbs',
            '192',
        ],
    )
    assert result.exit_code != 0
    assert 'not applicable' in result.output


def test_cli_no_plots_skips_figures(tmp_path):
    """--no-plots writes matrices and tables but no PNGs."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ['-i', MINTEST, '-d', str(tmp_path), '-p', 'mt', '--sbs', '96', '--no-plots'],
    )
    assert result.exit_code == 0, result.output
    produced = os.listdir(tmp_path)
    assert 'mt.SBS96.txt' in produced
    assert not any(name.endswith('.png') for name in produced)


def test_cli_partition_by_row(tmp_path):
    """--partition-by row produces one matrix column per sequence."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            MINTEST,
            '-d',
            str(tmp_path),
            '-p',
            'mt',
            '--sbs',
            '96',
            '--partition-by',
            'row',
            '--no-plots',
        ],
    )
    assert result.exit_code == 0, result.output
    _, samples, matrix = read_sbs_matrix(str(tmp_path / 'mt.SBS96.txt'))
    assert len(samples) == 6
    assert matrix.shape == (96, 6)


def test_group_lookup_tolerates_sanitised_names(tmp_path):
    """The groups lookup matches both original and IQ-TREE-sanitised names."""
    from derip2.app_spectra import _load_group_lookup

    mapping = tmp_path / 'groups.tsv'
    mapping.write_text('name\tgroup\nscf:1-9(+)\tspeciesA\nSeq2\tspeciesB\n')
    lookup = _load_group_lookup(str(mapping))
    # Original name resolves...
    assert lookup('scf:1-9(+)') == 'speciesA'
    # ...and so does the sanitised form IQ-TREE would write.
    assert lookup('scf_1-9___') == 'speciesA'
    assert lookup('Seq2') == 'speciesB'
    assert lookup('unknown') is None


def test_cli_groups_baseline(tmp_path):
    """--groups splits the baseline matrix into one column per group."""
    mapping = tmp_path / 'groups.tsv'
    mapping.write_text(
        'name\tgroup\nSeq1\tA\nSeq2\tA\nSeq3\tA\nSeq4\tB\nSeq5\tB\nSeq6\tB\n'
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            MINTEST,
            '-d',
            str(tmp_path),
            '-p',
            'g',
            '--sbs',
            '96',
            '--groups',
            str(mapping),
            '--no-plots',
        ],
    )
    assert result.exit_code == 0, result.output
    _, samples, matrix = read_sbs_matrix(str(tmp_path / 'g.SBS96.txt'))
    assert set(samples) == {'A', 'B'}
    assert matrix.shape == (96, 2)


@pytest.mark.skipif(not _HAVE_IQTREE, reason='IQ-TREE not on PATH')
def test_cli_groups_phylo(tmp_path):
    """--groups works for the phylo path, giving per-group sample columns."""
    mapping = tmp_path / 'groups.tsv'
    mapping.write_text(
        'name\tgroup\nSeq1\tA\nSeq2\tA\nSeq3\tA\nSeq4\tB\nSeq5\tB\nSeq6\tB\n'
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            MINTEST,
            '-d',
            str(tmp_path),
            '-p',
            'g',
            '--sbs',
            '96',
            '--method',
            'phylo',
            '--iqtree-model',
            'JC',
            '--threads',
            '1',
            '--groups',
            str(mapping),
            '--no-plots',
        ],
    )
    assert result.exit_code == 0, result.output
    _, samples, _ = read_sbs_matrix(str(tmp_path / 'g.SBS96.txt'))
    assert {'A', 'B'} & set(samples)


def test_cli_clade_partition_rejected_for_baseline(tmp_path):
    """--partition-by clade is only valid for --method phylo."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            MINTEST,
            '-d',
            str(tmp_path),
            '-p',
            'mt',
            '--partition-by',
            'clade',
            '--no-plots',
        ],
    )
    assert result.exit_code != 0


def _write_alignment(path, records):
    """Write ``(id, seq)`` pairs to a FASTA file."""
    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqIO import write
    from Bio.SeqRecord import SeqRecord

    aln = MultipleSeqAlignment(
        [SeqRecord(Seq(seq), id=rid, description='') for rid, seq in records]
    )
    write(aln, str(path), 'fasta')


def _mintest_records():
    """Return mintest as a list of ``(id, seq)`` pairs."""
    from Bio import SeqIO

    return [(rec.id, str(rec.seq)) for rec in SeqIO.parse(MINTEST, 'fasta')]


def test_reference_tag_detected_and_excluded(tmp_path):
    """A 'deRIPseq' row is used as ancestor and dropped from counted samples."""
    records = _mintest_records()
    width = len(records[0][1])
    aln_path = tmp_path / 'withref.fa'
    _write_alignment(aln_path, records + [('deRIPseq', 'A' * width)])

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            str(aln_path),
            '-d',
            str(tmp_path),
            '-p',
            'mt',
            '--sbs',
            '96',
            '--partition-by',
            'row',
            '--no-plots',
        ],
    )
    assert result.exit_code == 0, result.output
    _, samples, matrix = read_sbs_matrix(str(tmp_path / 'mt.SBS96.txt'))
    # The seven-row input yields six counted samples; the reference is excluded.
    assert len(samples) == 6
    assert 'deRIPseq' not in samples


def test_custom_reference_tag(tmp_path):
    """--reference-tag detects a differently-named reference row."""
    records = _mintest_records()
    width = len(records[0][1])
    aln_path = tmp_path / 'withref.fa'
    _write_alignment(aln_path, records + [('MyAnc', 'A' * width)])

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            str(aln_path),
            '-d',
            str(tmp_path),
            '-p',
            'mt',
            '--sbs',
            '96',
            '--partition-by',
            'row',
            '--reference-tag',
            'MyAnc',
            '--no-plots',
        ],
    )
    assert result.exit_code == 0, result.output
    _, samples, _ = read_sbs_matrix(str(tmp_path / 'mt.SBS96.txt'))
    assert len(samples) == 6
    assert 'MyAnc' not in samples


def test_ancestor_file_happy_path(tmp_path):
    """A single-sequence --ancestor FASTA of the right length is accepted."""
    records = _mintest_records()
    width = len(records[0][1])
    anc = tmp_path / 'anc.fa'
    anc.write_text(f'>anc\n{"A" * width}\n')

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            MINTEST,
            '-d',
            str(tmp_path),
            '-p',
            'mt',
            '--sbs',
            '96',
            '--ancestor',
            str(anc),
            '--no-plots',
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / 'mt.SBS96.txt').exists()


def test_ancestor_file_overrides_in_msa_reference(tmp_path):
    """When both are present, --ancestor wins and the deRIPseq row stays counted."""
    records = _mintest_records()
    width = len(records[0][1])
    aln_path = tmp_path / 'withref.fa'
    _write_alignment(aln_path, records + [('deRIPseq', 'A' * width)])
    anc = tmp_path / 'anc.fa'
    anc.write_text(f'>anc\n{"C" * width}\n')

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            str(aln_path),
            '-d',
            str(tmp_path),
            '-p',
            'mt',
            '--sbs',
            '96',
            '--ancestor',
            str(anc),
            '--partition-by',
            'row',
            '--no-plots',
        ],
    )
    assert result.exit_code == 0, result.output
    _, samples, _ = read_sbs_matrix(str(tmp_path / 'mt.SBS96.txt'))
    # The file overrides the in-MSA row, which is therefore NOT excluded.
    assert len(samples) == 7
    assert 'deRIPseq' in samples


def test_reference_exclusion_too_few_sequences(tmp_path):
    """Excluding the reference from a two-row alignment is a clear error."""
    records = _mintest_records()
    width = len(records[0][1])
    aln_path = tmp_path / 'tiny.fa'
    # One real sequence + the reference: excluding the reference leaves one row.
    _write_alignment(aln_path, [records[0], ('deRIPseq', 'A' * width)])

    runner = CliRunner()
    result = runner.invoke(
        main, ['-i', str(aln_path), '-d', str(tmp_path), '-p', 'mt', '--no-plots']
    )
    assert result.exit_code != 0
    assert 'fewer than 2 sequences' in result.output


def test_ancestor_length_mismatch_rejected(tmp_path):
    """An --ancestor of the wrong length is a clear CLI error."""
    anc = tmp_path / 'anc.fa'
    anc.write_text('>anc\nACGT\n')  # far shorter than the alignment

    runner = CliRunner()
    result = runner.invoke(
        main,
        ['-i', MINTEST, '-d', str(tmp_path), '-p', 'mt', '--ancestor', str(anc)],
    )
    assert result.exit_code != 0
    assert 'does not match the alignment width' in result.output


def test_degenerate_character_rejected(tmp_path):
    """An N in the alignment is rejected before any spectrum work."""
    records = _mintest_records()
    rid, seq = records[0]
    records[0] = (rid, 'N' + seq[1:])
    aln_path = tmp_path / 'withN.fa'
    _write_alignment(aln_path, records)

    runner = CliRunner()
    result = runner.invoke(
        main, ['-i', str(aln_path), '-d', str(tmp_path), '-p', 'mt', '--no-plots']
    )
    assert result.exit_code != 0
    assert 'non-ACGT character' in result.output


def test_lowercase_alignment_accepted(tmp_path):
    """Soft-masked (lower-case) input is accepted and normalised."""
    records = [(rid, seq.lower()) for rid, seq in _mintest_records()]
    aln_path = tmp_path / 'lower.fa'
    _write_alignment(aln_path, records)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            str(aln_path),
            '-d',
            str(tmp_path),
            '-p',
            'mt',
            '--sbs',
            '96',
            '--no-plots',
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / 'mt.SBS96.txt').exists()


def test_group_lookup_skips_blank_comment_and_short_lines(tmp_path):
    """Blank/comment/one-column lines are ignored; real pairs still resolve."""
    from derip2.app_spectra import _load_group_lookup

    mapping = tmp_path / 'groups.tsv'
    mapping.write_text(
        '# a comment\n'
        '\n'
        'orphan\n'  # only one column -> skipped
        'Seq1\tA\n'
        'Seq2\tB\n'
    )
    lookup = _load_group_lookup(str(mapping))
    assert lookup('Seq1') == 'A'
    assert lookup('Seq2') == 'B'
    assert lookup('orphan') is None


def test_group_lookup_empty_file_raises(tmp_path):
    """A file with no usable pairs is an error."""
    from derip2.app_spectra import _load_group_lookup

    mapping = tmp_path / 'groups.tsv'
    mapping.write_text('# only comments\n\n')
    with pytest.raises(ValueError, match='No sequence/group pairs'):
        _load_group_lookup(str(mapping))


@pytest.mark.skipif(not _HAVE_IQTREE, reason='IQ-TREE not on PATH')
def test_cli_phylo_path(tmp_path):
    """The phylo path runs IQ-TREE and writes QC, manifest and matrices."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            '-i',
            MINTEST,
            '-d',
            str(tmp_path),
            '-p',
            'ph',
            '--method',
            'phylo',
            '--iqtree-model',
            'JC',
            '--threads',
            '1',
            '--rooting',
            'outgroup',
            '--outgroup',
            'Seq1',
            '--root-sensitivity',
            '--no-plots',
        ],
    )
    assert result.exit_code == 0, result.output
    produced = set(os.listdir(tmp_path))
    for name in (
        'ph.SBS96.txt',
        'ph_qc_report.txt',
        'ph_column_gap_profile.tsv',
        'ph_run_manifest.json',
        'ph_events.tsv',
    ):
        assert name in produced, f'missing {name}'
    # The events table carries parent/child columns for the phylo path.
    with open(tmp_path / 'ph_events.tsv') as handle:
        header = handle.readline()
    assert 'parent' in header and 'child' in header
