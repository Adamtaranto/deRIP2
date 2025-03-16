import argparse
import os
import tempfile
from unittest.mock import patch

# Import the main function
from derip2.app import main


def test_main_function():
    """Test the main function with mintest.fa as input."""

    # Create temp directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define output filenames
        output_fasta = 'derip_output.fa'
        output_aln = 'derip_alignment.fa'

        # Create a mock argparse.Namespace object with all required arguments
        # This simulates what mainArgs() would return
        mock_args = argparse.Namespace(
            inAln='tests/data/mintest.fa',  # Path to our test file
            format='fasta',
            maxGaps=0.7,
            reaminate=True,
            maxSNPnoise=0.5,
            minRIPlike=0.1,
            fillindex=None,
            fillmaxgc=False,
            mask=True,
            noappend=False,
            outDir=temp_dir,
            outFasta=output_fasta,
            outAln=output_aln,
            outAlnFormat='fasta',
            label='TestDeRIP',
            loglevel='INFO',
            logfile=None,
        )

        # Patch mainArgs to return our mock_args
        with patch('derip2.app.mainArgs', return_value=mock_args):
            # Run the main function
            main()

            # Check that output files were created
            assert os.path.exists(os.path.join(temp_dir, output_fasta))
            assert os.path.exists(os.path.join(temp_dir, output_aln))

            # Check content of output FASTA file
            with open(os.path.join(temp_dir, output_fasta), 'r') as f:
                content = f.read()
                assert '>TestDeRIP' in content
                # Further content checks can be added

            # Check that alignment file has correct format and includes deRIPed sequence
            with open(os.path.join(temp_dir, output_aln), 'r') as f:
                content = f.read()
                assert '>TestDeRIP' in content
                assert '>Seq1' in content
