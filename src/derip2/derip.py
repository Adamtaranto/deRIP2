"""
DeRIP class for detecting and correcting RIP mutations in DNA alignments.

This module provides a class-based interface to the deRIP2 tool for correcting
Repeat-Induced Point (RIP) mutations in fungal DNA alignments.
"""

import logging
from os import path
from typing import Optional

import click
from Bio.Align import MultipleSeqAlignment

import derip2.aln_ops as ao


class DeRIP:
    """
    A class to detect and correct RIP (Repeat-Induced Point) mutations in DNA alignments.

    This class encapsulates the functionality to analyze DNA sequence alignments for
    RIP-like mutations, correct them, and generate deRIPed consensus sequences.

    Parameters
    ----------
    alignment_file : str
        Path to the alignment file in FASTA format.
    maxSNPnoise : float, optional
        Maximum proportion of conflicting SNPs permitted before excluding column
        from RIP/deamination assessment (default: 0.5).
    minRIPlike : float, optional
        Minimum proportion of deamination events in RIP context required for
        column to be deRIP'd in final sequence (default: 0.1).
    reaminate : bool, optional
        Whether to correct all deamination events independent of RIP context (default: False).
    fillindex : int, optional
        Index of row to use for filling uncorrected positions (default: None).
    fillmaxgc : bool, optional
        Whether to use sequence with highest GC content for filling if
        no row index is specified (default: False).
    maxGaps : float, optional
        Maximum proportion of gaps in a column before considering it a gap
        in consensus (default: 0.7).

    Attributes
    ----------
    alignment : MultipleSeqAlignment
        The loaded DNA sequence alignment.
    masked_alignment : MultipleSeqAlignment
        The alignment with RIP-corrected positions masked with IUPAC codes.
    consensus : SeqRecord
        The deRIPed consensus sequence.
    gapped_consensus : SeqRecord
        The deRIPed consensus sequence with gaps.
    rip_counts : Dict
        Dictionary tracking RIP mutation counts for each sequence.
    corrected_positions : Dict
        Dictionary of corrected positions {col_idx: {row_idx: {observed_base, corrected_base}}}.
    """

    def __init__(
        self,
        alignment_file: str,
        maxSNPnoise: float = 0.5,
        minRIPlike: float = 0.1,
        reaminate: bool = False,
        fillindex: Optional[int] = None,
        fillmaxgc: bool = False,
        maxGaps: float = 0.7,
    ) -> None:
        """
        Initialize DeRIP with an alignment file and parameters.

        Parameters
        ----------
        alignment_file : str
            Path to the alignment file in FASTA format.
        maxSNPnoise : float, optional
            Maximum proportion of conflicting SNPs permitted before excluding column
            from RIP/deamination assessment (default: 0.5).
        minRIPlike : float, optional
            Minimum proportion of deamination events in RIP context required for
            column to be deRIP'd in final sequence (default: 0.1).
        reaminate : bool, optional
            Whether to correct all deamination events independent of RIP context (default: False).
        fillindex : int, optional
            Index of row to use for filling uncorrected positions (default: None).
        fillmaxgc : bool, optional
            Whether to use sequence with highest GC content for filling if
            no row index is specified (default: False).
        maxGaps : float, optional
            Maximum proportion of gaps in a column before considering it a gap
            in consensus (default: 0.7).
        """
        # Store parameters
        self.maxSNPnoise = maxSNPnoise
        self.minRIPlike = minRIPlike
        self.reaminate = reaminate
        self.fillindex = fillindex
        self.fillmaxgc = fillmaxgc
        self.maxGaps = maxGaps

        # Initialize attributes
        self.alignment = None
        self.masked_alignment = None
        self.consensus = None
        self.gapped_consensus = None
        self.consensus_tracker = None
        self.rip_counts = None
        self.corrected_positions = {}
        self.colored_consensus = None

        # Load the alignment file
        self._load_alignment(alignment_file)

    def _load_alignment(self, alignment_file: str) -> None:
        """
        Load and validate the alignment file.

        Parameters
        ----------
        alignment_file : str
            Path to the alignment file.

        Raises
        ------
        FileNotFoundError
            If the alignment file does not exist.
        ValueError
            If the alignment contains fewer than two sequences or has duplicate IDs.
        """
        # Check if file exists
        if not path.isfile(alignment_file):
            raise FileNotFoundError(f'Alignment file not found: {alignment_file}')

        # Load alignment using aln_ops function
        try:
            self.alignment = ao.loadAlign(alignment_file, alnFormat='fasta')
            logging.info(f'Loaded alignment with {len(self.alignment)} sequences')
        except Exception as e:
            raise ValueError(f'Error loading alignment: {str(e)}') from e

    def calculate_rip(self, label: str = 'deRIPseq') -> None:
        """
        Calculate RIP locations and corrections in the alignment.

        This method performs RIP detection and correction, fills in the consensus
        sequence, and populates the class attributes.

        Parameters
        ----------
        label : str, optional
            ID for the generated deRIPed sequence (default: "deRIPseq").

        Returns
        -------
        None
            Updates class attributes with results.
        """
        # Initialize tracking structures
        # tracker is a dict of tuples, keys are column indices, values are tuples of (col_idx, corrected_base)
        # used to compose the consensus sequence
        tracker = ao.initTracker(self.alignment)
        # rip_counts is a dict of rowItem('idx', 'SeqID', 'revRIPcount', 'RIPcount', 'nonRIPcount', 'GC'), keys are row IDs
        # used to track RIP mutations in each sequence
        rip_counts = ao.initRIPCounter(self.alignment)

        # Pre-fill conserved positions
        tracker = ao.fillConserved(self.alignment, tracker, self.maxGaps)

        # Detect and correct RIP mutations
        tracker, rip_counts, masked_alignment, corrected_positions = ao.correctRIP(
            self.alignment,
            tracker,
            rip_counts,
            maxSNPnoise=self.maxSNPnoise,
            minRIPlike=self.minRIPlike,
            reaminate=self.reaminate,
            mask=True,  # Always mask so we have the masked alignment available
        )

        # Populate corrected positions dictionary
        self._build_corrected_positions(self.alignment, masked_alignment)

        # Select reference sequence for filling uncorrected positions
        if self.fillindex is not None:
            # Validate index is within range
            ao.checkrow(self.alignment, idx=self.fillindex)
            ref_id = self.fillindex
        else:
            # Select based on RIP counts or GC content
            ref_id = ao.setRefSeq(
                self.alignment,
                rip_counts,
                getMinRIP=not self.fillmaxgc,  # Use sequence with fewest RIPs if not filling with max GC
                getMaxGC=self.fillmaxgc,
            )

        # Fill remaining positions from selected reference sequence
        tracker = ao.fillRemainder(self.alignment, ref_id, tracker)

        # Create consensus sequence
        consensus = ao.getDERIP(tracker, ID=label, deGAP=True)
        gapped_consensus = ao.getDERIP(tracker, ID=label, deGAP=False)

        # Store results in attributes
        self.masked_alignment = masked_alignment
        self.consensus = consensus
        self.gapped_consensus = gapped_consensus
        self.consensus_tracker = tracker
        self.rip_counts = rip_counts

        # Create colorized consensus
        self._colorize_corrected_positions()

        # Log summary
        logging.info(
            f'RIP correction complete. Reference sequence used for filling: {ref_id}'
        )

    def _build_corrected_positions(
        self, original: MultipleSeqAlignment, masked: MultipleSeqAlignment
    ) -> None:
        """
        Build dictionary of corrected positions by comparing original and masked alignments.

        Parameters
        ----------
        original : MultipleSeqAlignment
            The original alignment.
        masked : MultipleSeqAlignment
            The masked alignment with RIP positions marked.

        Returns
        -------
        None
            Updates the corrected_positions attribute.
        """
        self.corrected_positions = {}

        # Compare original and masked alignments
        for col_idx in range(original.get_alignment_length()):
            col_dict = {}

            for row_idx in range(len(original)):
                orig_base = original[row_idx].seq[col_idx]
                masked_base = masked[row_idx].seq[col_idx]

                # Check if this position was masked (corrected)
                if orig_base != masked_base:
                    # Determine the corrected base based on the IUPAC code
                    corrected_base = None
                    if masked_base == 'Y':  # C or T (C→T)
                        corrected_base = 'C'
                    elif masked_base == 'R':  # G or A (G→A)
                        corrected_base = 'G'

                    if corrected_base:
                        col_dict[row_idx] = {
                            'observed_base': orig_base,
                            'corrected_base': corrected_base,
                        }

            # Only add column to dict if corrections were made
            if col_dict:
                self.corrected_positions[col_idx] = col_dict

        logging.info(
            f'Identified {len(self.corrected_positions)} columns with RIP corrections'
        )

    def _colorize_corrected_positions(self) -> str:
        """
        Create a colorized version of the gapped consensus sequence.

        Bases at positions that were corrected during RIP analysis
        are highlighted in green.

        Returns
        -------
        str
            Consensus sequence with corrected positions highlighted in green.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.gapped_consensus is None:
            raise ValueError('Must call calculate_rip before colorizing consensus')

        # Get the consensus sequence as a string
        seq_str = str(self.gapped_consensus.seq)

        # Convert to list for easier manipulation
        seq_chars = list(seq_str)

        # Add ANSI color codes for each corrected position
        BOLD_GREEN = '\033[1;32m'  # 1 for bold, 32 for green
        RESET = '\033[0m'

        for pos in self.corrected_positions:
            if 0 <= pos < len(seq_chars):
                # Only colorize if position is in range (safety check)
                seq_chars[pos] = f'{BOLD_GREEN}{seq_chars[pos]}{RESET}'

        # Join back into string
        colored_seq = ''.join(seq_chars)

        # Store as attribute
        self.colored_consensus = colored_seq

        return colored_seq

    def write_alignment(
        self,
        output_file: str,
        append_consensus: bool = True,
        mask_rip: bool = True,
        consensus_id: str = 'deRIPseq',
        format: str = 'fasta',
    ) -> None:
        """
        Write alignment to file with options to append consensus and mask RIP positions.

        Parameters
        ----------
        output_file : str
            Path to the output alignment file.
        append_consensus : bool, optional
            Whether to append the consensus sequence to the alignment (default: True).
        mask_rip : bool, optional
            Whether to mask RIP positions in the output alignment (default: True).
        consensus_id : str, optional
            ID for the consensus sequence if appended (default: "deRIPseq").
        format : str, optional
            Format for the output alignment file (default: "fasta").

        Returns
        -------
        None
            Writes alignment to file.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.consensus_tracker is None:
            raise ValueError('Must call calculate_rip before writing output')

        # Select alignment based on masking preference
        source_alignment = self.masked_alignment if mask_rip else self.alignment

        # Write the alignment file
        ao.writeAlign(
            self.consensus_tracker,
            source_alignment,
            output_file,
            ID=consensus_id,
            outAlnFormat=format,
            noappend=not append_consensus,
        )

        logging.info(f'Alignment written to {output_file}')

    def write_consensus(self, output_file: str, consensus_id: str = 'deRIPseq') -> None:
        """
        Write the deRIPed consensus sequence to a FASTA file.

        Parameters
        ----------
        output_file : str
            Path to the output FASTA file.
        consensus_id : str, optional
            ID for the consensus sequence (default: "deRIPseq").

        Returns
        -------
        None
            Writes consensus sequence to file.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.consensus_tracker is None:
            raise ValueError('Must call calculate_rip before writing output')

        # Write the sequence to file
        ao.writeDERIP(self.consensus_tracker, output_file, ID=consensus_id)

        logging.info(f'Consensus sequence written to {output_file}')

    def get_consensus_string(self) -> str:
        """
        Get the deRIPed consensus sequence as a string.

        Returns
        -------
        str
            The deRIPed consensus sequence.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.consensus is None:
            raise ValueError('Must call calculate_rip before accessing consensus')

        return str(self.consensus.seq)

    def print_rip_summary(self) -> None:
        """
        Print a summary of RIP mutations found in each sequence.

        Returns
        -------
        None
            Prints summary to standard error.

        Raises
        ------
        ValueError
            If calculate_rip has not been called first.
        """
        if self.rip_counts is None:
            raise ValueError('Must call calculate_rip before printing RIP summary')

        ao.summarizeRIP(self.rip_counts)


def get_derip_consensus(
    input_file: str,
    output_file: str,
    consensus_name: str = 'derip_consensus',
    maxSNPnoise: float = 0.5,
    minRIPlike: float = 0.1,
    maxGaps: float = 0.7,
    reaminate: bool = False,
    fillindex: Optional[int] = None,
    fillmaxgc: bool = False,
):
    """
    Generate a deRIPed consensus sequence from an alignment file.

    This function processes a DNA alignment file to identify and correct Repeat-Induced
    Point (RIP) mutations, then writes the corrected consensus sequence to an output file.
    It also prints a summary of identified RIP mutations and alignment information.

    Parameters
    ----------
    input_file : str
        Path to the input alignment file in FASTA format.
    output_file : str
        Path where the consensus sequence will be written.
    consensus_name : str, optional
        Name for the consensus sequence in the output file (default: 'derip_consensus').
    maxSNPnoise : float, optional
        Maximum proportion of conflicting SNPs permitted before excluding a column
        from RIP/deamination assessment (default: 0.5).
    minRIPlike : float, optional
        Minimum proportion of deamination events in RIP context required for a
        column to be deRIPed in final sequence (default: 0.1).
    maxGaps : float, optional
        Maximum proportion of gaps in a column before considering it a gap
        in the consensus sequence (default: 0.7).
    reaminate : bool, optional
        Whether to correct all deamination events independent of RIP context (default: False).
    fillindex : int, optional
        Index of row to use for filling uncorrected positions (default: None).
    fillmaxgc : bool, optional
        Whether to use the sequence with highest GC content for filling if
        no row index is specified (default: False).

    Returns
    -------
    None
        This function writes to a file and prints to stdout but doesn't return a value.

    Raises
    ------
    FileNotFoundError
        If the specified input file does not exist.

    Notes
    -----
    The function prints summary information to standard output, including:
    - Number of columns repaired
    - RIP mutation summary by sequence
    - Visualization of the masked alignment
    - Gapped consensus sequence
    """
    if path.isfile(input_file):
        derip_object = DeRIP(
            input_file,
            maxSNPnoise=maxSNPnoise,
            minRIPlike=minRIPlike,
            maxGaps=maxGaps,
            reaminate=reaminate,
            fillindex=fillindex,
            fillmaxgc=fillmaxgc,
        )
        # Calculate RIP mutations
        derip_object.calculate_rip(label=consensus_name)

        # Access corrected positions
        print(
            f'DeRIP2 found {len(derip_object.corrected_positions)} columns to be repaired.\n'
        )

        # Print RIP summary
        print('RIP summary by row:')
        derip_object.print_rip_summary()

        # Print raw alignment
        print(f'\nRaw alignment:\n{derip_object.alignment}')
        print(f'{derip_object.colored_consensus} {consensus_name}\n')

        # Print masked alignment
        print('\nMutation masked alignment:\n', derip_object.masked_alignment)
        print(f'{derip_object.colored_consensus} {consensus_name}\n')

        # Opt1: Write output consensus to file
        derip_object.write_consensus(output_file, consensus_id=consensus_name)

        # Opt2: Write original alignment with appened deRIP'd sequence to output file
        # derip_object.write_alignment(output_file, append_consensus=True, mask_rip=False)

    else:
        raise FileNotFoundError(f"The file '{input_file}' does not exist.")


@click.command()
@click.option(
    '--input_file',
    '-i',
    required=True,
    type=str,
    help='Multiple sequence alignment FASTA file path',
)
@click.option('--output_file', '-o', required=True, type=str, help='Output file')
@click.option(
    '--consensus_name',
    '-n',
    type=str,
    default='derip_consensus',
    help='Name of the consensus sequence (default: derip_consensus)',
)
@click.option(
    '--maxSNPnoise',
    type=float,
    default=0.5,
    help='Maximum proportion of conflicting SNPs permitted before excluding column from RIP/deamination assessment (default: 0.5)',
)
@click.option(
    '--minRIPlike',
    type=float,
    default=0.1,
    help='Minimum proportion of deamination events in RIP context required for column to be deRIPd in final sequence (default: 0.1)',
)
@click.option(
    '--maxGaps',
    type=float,
    default=0.7,
    help='Maximum proportion of gaps in a column before considering it a gap in consensus (default: 0.7)',
)
@click.option(
    '--reaminate',
    is_flag=True,
    help='Correct all deamination events independent of RIP context',
)
@click.option(
    '--fillindex',
    type=int,
    help='Index of row to use for filling uncorrected positions',
)
@click.option(
    '--fillmaxgc',
    is_flag=True,
    help='Use sequence with highest GC content for filling if no row index is specified',
)
def derip_click(
    input_file,
    output_file,
    consensus_name,
    maxsnpnoise,
    minriplike,
    maxgaps,
    reaminate,
    fillindex,
    fillmaxgc,
):
    """
    Command line interface for the deRIP consensus generation tool.

    This function serves as the entry point for the command line interface,
    processing arguments from Click decorators and passing them to the
    get_derip_consensus function.

    Parameters
    ----------
    input_file : str
        Path to the input alignment file in FASTA format.
    output_file : str
        Path where the consensus sequence will be written.
    consensus_name : str
        Name for the consensus sequence in the output file.
    maxsnpnoise : float
        Maximum proportion of conflicting SNPs permitted before excluding a column
        from RIP/deamination assessment.
    minriplike : float
        Minimum proportion of deamination events in RIP context required for a
        column to be deRIPed in final sequence.
    maxgaps : float
        Maximum proportion of gaps in a column before considering it a gap
        in the consensus sequence.
    reaminate : bool
        Whether to correct all deamination events independent of RIP context.
    fillindex : int or None
        Index of row to use for filling uncorrected positions.
    fillmaxgc : bool
        Whether to use the sequence with highest GC content for filling if
        no row index is specified.

    Returns
    -------
    None
        This function calls get_derip_consensus which writes output to a file
        and prints information to stdout.

    Notes
    -----
    This function is intended to be used with Click as a command-line entry point
    and should not typically be called directly in code.
    """
    get_derip_consensus(
        input_file=input_file,
        output_file=output_file,
        consensus_name=consensus_name,
        maxSNPnoise=maxsnpnoise,
        minRIPlike=minriplike,
        maxGaps=maxgaps,
        reaminate=reaminate,
        fillindex=fillindex,
        fillmaxgc=fillmaxgc,
    )


if __name__ == '__main__':
    derip_click()
    # Example usage:
    # python -m src.derip2.derip -i tests/data/mintest.fa -o outalign.fa -n min_element_derip --reaminate
