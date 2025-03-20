import os
import tempfile

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
                '--format',
                'fasta',
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
                '--format',
                'fasta',
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
        output_viz = os.path.join(temp_dir, f'{prefix}_visualization.png')

        # Verify files exist
        assert os.path.exists(output_fasta), (
            f'Output FASTA file not found: {output_fasta}'
        )
        assert os.path.exists(output_aln), (
            f'Output alignment file not found: {output_aln}'
        )
        assert os.path.exists(output_viz), f'Visualization file not found: {output_viz}'


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
                '--format',
                'fasta',
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
