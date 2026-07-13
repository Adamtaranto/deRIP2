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
