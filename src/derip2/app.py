#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
██████╗ ███████╗██████╗ ██╗██████╗ ██████╗
██╔══██╗██╔════╝██╔══██╗██║██╔══██╗╚════██╗
██║  ██║█████╗  ██████╔╝██║██████╔╝ █████╔╝
██║  ██║██╔══╝  ██╔══██╗██║██╔═══╝ ██╔═══╝
██████╔╝███████╗██║  ██║██║██║     ███████╗
╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚══════╝.

Takes a multi-sequence DNA alignment and estimates a progenitor sequence by
correcting for RIP-like mutations. deRIP2 searches all available sequences for
evidence of un-RIP'd precursor states at each aligned position, allowing for
improved RIP-correction across large repeat families in which members are
independently RIP'd.
"""

import logging
from os import path
import sys

import click

from derip2._version import __version__
import derip2.aln_ops as ao
from derip2.derip import DeRIP
from derip2.utils.checks import dochecks
from derip2.utils.logs import colored, init_logging


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Predict ancestral sequence of fungal repeat elements by correcting for RIP-like mutations or cytosine deamination in multi-sequence DNA alignments. Optionally, mask mutated positions in alignment.',
)
@click.version_option(version=__version__, prog_name='derip2')
# Input options
@click.option(
    '-i', '--input', required=True, type=str, help='Multiple sequence alignment.'
)
@click.option(
    '--format',
    type=click.Choice(['clustal', 'emboss', 'fasta', 'nexus', 'stockholm']),
    default='fasta',
    show_default=True,
    help='Format of input alignment.',
)
# Algorithm parameters
@click.option(
    '-g',
    '--max-gaps',
    type=float,
    default=0.7,
    show_default=True,
    help='Maximum proportion of gapped positions in column to be tolerated before forcing a gap in final deRIP sequence.',
)
@click.option(
    '-a',
    '--reaminate',
    is_flag=True,
    default=False,
    show_default=True,
    help='Correct all deamination events independent of RIP context.',
)
@click.option(
    '--max-snp-noise',
    type=float,
    default=0.5,
    show_default=True,
    help="Maximum proportion of conflicting SNPs permitted before excluding column from RIP/deamination assessment. i.e. By default a column with >= 0.5 'C/T' bases will have 'TpA' positions logged as RIP events.",
)
@click.option(
    '--min-rip-like',
    type=float,
    default=0.1,
    show_default=True,
    help="Minimum proportion of deamination events in RIP context (5' CpA 3' --> 5' TpA 3') required for column to deRIP'd in final sequence. Note: If 'reaminate' option is set all deamination events will be corrected.",
)
# Reference sequence selection options
@click.option(
    '--fill-max-gc',
    is_flag=True,
    default=False,
    show_default=True,
    help='By default uncorrected positions in the output sequence are filled from the sequence with the lowest RIP count. If this option is set remaining positions are filled from the sequence with the highest G/C content.',
)
@click.option(
    '--fill-index',
    type=int,
    default=None,
    help="Force selection of alignment row to fill uncorrected positions from by row index number (indexed from 0). Note: Will override '--fill-max-gc' option.",
)
# Masking and output alignment options
@click.option(
    '--mask',
    is_flag=True,
    default=False,
    show_default=True,
    help='Mask corrected positions in alignment with degenerate IUPAC codes.',
)
@click.option(
    '--no-append',
    is_flag=True,
    default=False,
    show_default=True,
    help="If set, do not append deRIP'd sequence to output alignment.",
)
# Output file options
@click.option(
    '-d',
    '--out-dir',
    type=str,
    default=None,
    help="Directory for deRIP'd sequence files to be written to.",
)
@click.option(
    '-p',
    '--prefix',
    default='deRIPseq',
    show_default=True,
    help='Prefix for output files. Output files will be named prefix.fasta, prefix_alignment.fasta, etc.',
)
# Visualization options
@click.option(
    '--plot',
    is_flag=True,
    default=False,
    show_default=True,
    help='Create a visualization of the alignment with RIP markup.',
)
@click.option(
    '--plot-rip-type',
    type=click.Choice(['both', 'product', 'substrate']),
    default='both',
    show_default=True,
    help='Specify the type of RIP events to be displayed in the alignment visualization.',
)
# Logging options
@click.option(
    '--loglevel',
    type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
    default='INFO',
    show_default=True,
    help='Set logging level.',
)
@click.option('--logfile', default=None, help='Log file path.')
def main(
    input,
    format,
    max_gaps,
    reaminate,
    max_snp_noise,
    min_rip_like,
    fill_max_gc,
    fill_index,
    mask,
    no_append,
    out_dir,
    prefix,
    plot,
    plot_rip_type,
    loglevel,
    logfile,
):
    """
    Main execution function for deRIP2.

    This function coordinates the entire deRIP workflow:
    1. Processes command line arguments
    2. Sets up logging and output directories
    3. Loads and validates the input alignment
    4. Performs RIP detection and correction
    5. Fills in remaining positions from a reference sequence
    6. Generates output files including the deRIPed sequence and optionally a
       masked alignment.

    Parameters
    ----------
    input : str
        Path to multiple sequence alignment file.
    format : str
        Format of input alignment. One of: 'clustal', 'emboss', 'fasta', 'nexus',
        or 'stockholm'. Default: 'fasta'.
    max_gaps : float
        Maximum proportion of gapped positions in column to be tolerated before
        forcing a gap in final deRIP sequence. Default: 0.7.
    reaminate : bool
        If True, correct all deamination events independent of RIP context.
        Default: False.
    max_snp_noise : float
        Maximum proportion of conflicting SNPs permitted before excluding column
        from RIP/deamination assessment. Default: 0.5.
    min_rip_like : float
        Minimum proportion of deamination events in RIP context required for column
        to be deRIP'd in final sequence. Default: 0.1.
    fill_max_gc : bool
        If True, fill uncorrected positions from the sequence with the highest G/C content
        rather than the least RIP'd sequence. Default: False.
    fill_index : int or None
        If provided, force selection of alignment row to fill uncorrected positions
        from by row index number (indexed from 0). Overrides 'fill_max_gc' option.
        Default: None.
    mask : bool
        If True, mask corrected positions in alignment with degenerate IUPAC codes.
        Default: False.
    no_append : bool
        If True, do not append deRIP'd sequence to output alignment. Default: False.
    out_dir : str or None
        Directory for deRIP'd sequence files to be written to. If None, uses current directory.
        Default: None.
    prefix : str
        Prefix for output files. Output files will be named prefix.fasta,
        prefix_alignment.fasta, etc. Default: 'deRIPseq'.
    plot : bool
        If True, create a visualization of the alignment with RIP markup.
        Default: False.
    plot_rip_type : str
        Specify the type of RIP events to be displayed in the alignment visualization.
        One of: 'both', 'product', or 'substrate'. Default: 'both'.
    loglevel : str
        Set logging level. One of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
        Default: 'INFO'.
    logfile : str or None
        Log file path. If None, logs to console only. Default: None.

    Returns
    -------
    None
        Does not return any values, but writes output files and logs to the console.
    """
    # ---------- Setup ----------
    # Print full command line call
    print(f'Command line call: {colored.green(" ".join(sys.argv))}\n')

    # Check/create output directory
    out_dir, logfile = dochecks(out_dir, logfile)

    # Set up logging based on specified level
    init_logging(loglevel=loglevel, logfile=logfile)

    # Set standardized output file paths
    out_path_fasta = path.join(out_dir, f'{prefix}.fasta')
    out_path_aln = path.join(out_dir, f'{prefix}_alignment.fasta')
    if mask:
        out_path_aln = path.join(out_dir, f'{prefix}_masked_alignment.fasta')
    # Path for visualization - only used if plot is True
    viz_path = path.join(out_dir, f'{prefix}_visualization.png')

    # ---------- Create DeRIP object and process alignment ----------
    logging.info(f'Processing alignment file: \033[0m{input}')

    # Create DeRIP object with command line parameters
    derip_obj = DeRIP(
        alignment_input=input,
        maxSNPnoise=max_snp_noise,
        minRIPlike=min_rip_like,
        reaminate=reaminate,
        fillindex=fill_index,
        fillmaxgc=fill_max_gc,
        maxGaps=max_gaps,
    )

    # Report alignment summary
    logging.info(f'Loaded alignment with {len(derip_obj.alignment)} sequences')
    ao.alignSummary(derip_obj.alignment)

    # Calculate RIP mutations and generate consensus
    logging.info('Processing alignment for RIP mutations...')
    derip_obj.calculate_rip(label=prefix)

    # Access corrected positions
    logging.info(
        f'\nDeRIP2 found {len(derip_obj.corrected_positions)} columns to be repaired.\n'
    )

    # Print RIP summary
    logging.info(f'RIP summary by row:\n\033[0m{derip_obj.rip_summary()}\n')

    # Print colourized alignment + consensus
    logging.info(f'Corrected alignment:\n\033[0m{derip_obj}\n')

    # ---------- Output Results ----------
    # Report deRIP'd sequence to stdout
    logging.info(f'Final RIP corrected sequence: \033[0m{derip_obj.colored_consensus}')

    # Write deRIP'd sequence to FASTA file
    logging.info(f"Writing deRIP'd sequence to file: \033[0m{out_path_fasta}")
    derip_obj.write_consensus(out_path_fasta, consensus_id=prefix)

    # Write alignment file with deRIP'd sequence
    logging.info('Preparing output alignment.')

    # Log if deRIP'd sequence will be appended to alignment
    if not no_append:
        logging.info(
            f'Appending corrected sequence to alignment with ID: \033[0m{prefix}'
        )

    # Write the alignment to file
    logging.info(f'Writing alignment to path: \033[0m{out_path_aln}')
    derip_obj.write_alignment(
        output_file=out_path_aln,
        append_consensus=not no_append,
        mask_rip=mask,
        consensus_id=prefix,
        format='fasta',
    )

    # Create visualization highlighting RIP/deamination events if requested
    if plot:
        logging.info(
            f'Creating alignment visualization with RIP markup at: \033[0m{viz_path}'
        )

        # Get alignment dimensions for visualization options
        ali_height = len(derip_obj.alignment)
        ali_length = derip_obj.alignment.get_alignment_length()

        # Create the visualization
        viz_result = derip_obj.plot_alignment(
            output_file=viz_path,
            title=f'DeRIP2 Alignment: {prefix}',
            show_chars=(ali_height <= 25),  # Show characters only for small alignments
            draw_boxes=(
                ali_height <= 25
            ),  # Draw boxes around characters for small alignments
            show_rip=plot_rip_type,
            highlight_corrected=True,
            flag_corrected=(
                ali_length < 200
            ),  # Flag corrected positions for small alignments
        )

        if viz_result:
            logging.info(f'RIP visualization created at: \033[0m{viz_path}')
        else:
            logging.warning('Failed to create RIP visualization')


if __name__ == '__main__':
    main()
