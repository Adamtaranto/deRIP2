"""
CLI tests for the ``derip2-spectra`` entry point.

These drive the command through Click's ``CliRunner`` and check that the matrix
files, tables and figures are produced and are well-formed.
"""

import logging
import os

from click.testing import CliRunner

from derip2.app_spectra import main
from derip2.spectra import read_sbs_matrix

logging.disable(logging.CRITICAL)

HERE = os.path.dirname(__file__)
MINTEST = os.path.join(HERE, 'data', 'mintest.fa')


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
