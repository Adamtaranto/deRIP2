import os
import tempfile

import pytest
from click.testing import CliRunner

# Import the main function
from derip2.app import main


def test_main_function():
    """Test the main function with mintest.fa as input."""

    # Create temp directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up the Click test runner
        runner = CliRunner()

        # Define test parameters
        prefix = 'TestDeRIP'

        # Run the command using the Click test runner
        result = runner.invoke(
            main,
            [
                '-i',
                'tests/data/mintest.fa',
                '--max-gaps',
                '0.7',
                '--reaminate',
                '--max-snp-noise',
                '0.5',
                '--min-rip-like',
                '0.1',
                '--mask',
                '--out-dir',
                temp_dir,
                '--prefix',
                prefix,
                '--loglevel',
                'INFO',
            ],
        )

        # Check that command completed successfully
        assert result.exit_code == 0, f'Command failed with output: {result.output}'

        # Check that output files were created with standardized names
        output_fasta = os.path.join(temp_dir, f'{prefix}.fasta')
        output_aln = os.path.join(temp_dir, f'{prefix}_masked_alignment.fasta')

        # Verify files exist
        assert os.path.exists(output_fasta), (
            f'Output FASTA file not found: {output_fasta}'
        )
        assert os.path.exists(output_aln), (
            f'Output alignment file not found: {output_aln}'
        )

        # Check content of output FASTA file
        with open(output_fasta, 'r') as f:
            content = f.read()
            assert f'>{prefix}' in content
            # Further content checks can be added

        # Check that alignment file has correct format and includes deRIPed sequence
        with open(output_aln, 'r') as f:
            content = f.read()
            assert f'>{prefix}' in content
            assert '>Seq1' in content


def test_main_function_with_visualization():
    """Test the main function with visualization enabled."""

    # Create temp directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up the Click test runner
        runner = CliRunner()

        # Define test parameters
        prefix = 'TestDeRIP'

        # Run the command using the Click test runner with plot option enabled
        result = runner.invoke(
            main,
            [
                '-i',
                'tests/data/mintest.fa',
                '--max-gaps',
                '0.7',
                '--reaminate',
                '--max-snp-noise',
                '0.5',
                '--min-rip-like',
                '0.1',
                '--mask',
                '--out-dir',
                temp_dir,
                '--prefix',
                prefix,
                '--plot',
                '--plot-rip-type',
                'both',
                '--loglevel',
                'INFO',
            ],
        )

        # Check that command completed successfully
        assert result.exit_code == 0, f'Command failed with output: {result.output}'

        # Check that output files were created with standardized names
        output_fasta = os.path.join(temp_dir, f'{prefix}.fasta')
        output_aln = os.path.join(temp_dir, f'{prefix}_masked_alignment.fasta')
        output_viz = os.path.join(temp_dir, f'{prefix}_visualization.svg')

        # Verify files exist
        assert os.path.exists(output_fasta), (
            f'Output FASTA file not found: {output_fasta}'
        )
        assert os.path.exists(output_aln), (
            f'Output alignment file not found: {output_aln}'
        )
        assert os.path.exists(output_viz), f'Visualization file not found: {output_viz}'


@pytest.mark.parametrize('plot_format', ['svg', 'png'])
def test_plot_format_option(plot_format):
    """The --plot-format option controls the visualization file extension.

    ``svg`` keeps the default vector output; ``png`` writes a fully rasterised
    image (sharp at any zoom). ``drawMiniAlignment`` derives the save format from
    the output extension, so selecting the format here must produce a matching file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = CliRunner()
        prefix = 'FmtDeRIP'
        result = runner.invoke(
            main,
            [
                '-i',
                'tests/data/mintest.fa',
                '--out-dir',
                temp_dir,
                '--prefix',
                prefix,
                '--plot',
                '--plot-format',
                plot_format,
            ],
        )
        assert result.exit_code == 0, f'Command failed with output: {result.output}'

        expected = os.path.join(temp_dir, f'{prefix}_visualization.{plot_format}')
        assert os.path.exists(expected), f'Visualization not found: {expected}'
        # The other-format file must NOT be produced.
        other = 'png' if plot_format == 'svg' else 'svg'
        assert not os.path.exists(
            os.path.join(temp_dir, f'{prefix}_visualization.{other}')
        )
        if plot_format == 'png':
            # PNG magic bytes: a real rasterised image, not an SVG mislabelled.
            with open(expected, 'rb') as handle:
                assert handle.read(8) == b'\x89PNG\r\n\x1a\n'


def test_spectra_ref_index_option():
    """--spectra-ref-index selects an alternative spectra reference; bad values fail.

    A valid row index produces the per-sequence report and names that reference in
    the spectra prose; an out-of-range index is rejected as a bad CLI parameter.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = CliRunner()
        prefix = 'RefDeRIP'
        result = runner.invoke(
            main,
            [
                '-i',
                'tests/data/mintest.fa',
                '--out-dir',
                temp_dir,
                '--prefix',
                prefix,
                '--per-seq-report',
                '--spectra-ref-index',
                '0',
            ],
        )
        assert result.exit_code == 0, f'Command failed with output: {result.output}'
        report = os.path.join(temp_dir, f'{prefix}_per_sequence.html')
        assert os.path.exists(report)
        assert 'reference sequence' in open(report).read()

        # Out-of-range index is rejected without writing a broken report.
        bad = runner.invoke(
            main,
            [
                '-i',
                'tests/data/mintest.fa',
                '--out-dir',
                temp_dir,
                '--prefix',
                'BadRef',
                '--per-seq-report',
                '--spectra-ref-index',
                '99',
            ],
        )
        assert bad.exit_code != 0
        assert 'out of range' in bad.output


def test_noappend_option():
    """Test that the --no-append option works correctly."""

    # Create temp directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up the Click test runner
        runner = CliRunner()

        # Define test parameters
        prefix = 'TestDeRIP'

        # Run the command with --no-append flag
        result = runner.invoke(
            main,
            [
                '-i',
                'tests/data/mintest.fa',
                '--no-append',
                '--out-dir',
                temp_dir,
                '--prefix',
                prefix,
                '--loglevel',
                'INFO',
            ],
        )

        # Check that command completed successfully
        assert result.exit_code == 0, f'Command failed with output: {result.output}'

        # Check the alignment file
        output_aln = os.path.join(temp_dir, f'{prefix}_alignment.fasta')

        # Verify file exists
        assert os.path.exists(output_aln), (
            f'Output alignment file not found: {output_aln}'
        )

        # Check that alignment file does NOT include the deRIPed sequence
        with open(output_aln, 'r') as f:
            content = f.read()
            assert f'>{prefix}' not in content, (
                'DeRIP sequence found in alignment despite --no-append flag'
            )
