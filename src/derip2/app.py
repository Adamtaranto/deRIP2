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

import argparse
import logging
from os import path
import sys

from argparse_tui import add_tui_argument

from derip2._version import __version__
import derip2.aln_ops as ao
from derip2.utils.checks import dochecks
from derip2.utils.logs import colored, init_logging


def mainArgs() -> argparse.Namespace:
    """
    Parse command line arguments for deRIP2.

    This function sets up the command line interface for deRIP2, defining all available
    options and their default values. The function handles input alignment options,
    RIP correction parameters, reference sequence selection, output formatting, and
    file writing options.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    # Create main parser with program description
    parser = argparse.ArgumentParser(
        description='Predict ancestral sequence of fungal repeat elements by correcting for RIP-like mutations or cytosine deamination in multi-sequence DNA alignments. Optionally, mask corrected positions in alignment.',
        prog='derip2',
    )

    # General program options
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {version}'.format(version=__version__),
    )
    # Input options
    parser.add_argument(
        '-i',
        '--inAln',
        required=True,
        type=str,
        default=None,
        help='Multiple sequence alignment.',
    )
    parser.add_argument(
        '--format',
        default='fasta',
        choices=[
            'clustal',
            'emboss',
            'fasta',
            'nexus',
            'stockholm',
        ],
        help='Format of input alignment. Default: fasta',
    )

    # Algorithm parameters
    parser.add_argument(
        '-g',
        '--maxGaps',
        type=float,
        default=0.7,
        help='Maximum proportion of gapped positions in column to be tolerated before forcing a gap in final deRIP sequence. Default: 0.7',
    )
    parser.add_argument(
        '-a',
        '--reaminate',
        action='store_true',
        default=False,
        help='Correct all deamination events independent of RIP context. Default: False',
    )
    parser.add_argument(
        '--maxSNPnoise',
        type=float,
        default=0.5,
        help="Maximum proportion of conflicting SNPs permitted before excluding column from RIP/deamination assessment. i.e. By default a column with >= 0.5 'C/T' bases will have 'TpA' positions logged as RIP events. Default: 0.5",
    )
    parser.add_argument(
        '--minRIPlike',
        type=float,
        default=0.1,
        help="Minimum proportion of deamination events in RIP context (5' CpA 3' --> 5' TpA 3') required for column to deRIP'd in final sequence. Note: If 'reaminate' option is set all deamination events will be corrected. Default 0.1 ",
    )

    # Reference sequence selection options
    parser.add_argument(
        '--fillmaxgc',
        action='store_true',
        default=False,
        help='By default uncorrected positions in the output sequence are filled from the sequence with the lowest RIP count. If this option is set remaining positions are filled from the sequence with the highest G/C content. Default: False',
    )
    parser.add_argument(
        '--fillindex',
        type=int,
        default=None,
        help="Force selection of alignment row to fill uncorrected positions from by row index number (indexed from 0). Note: Will override '--fillmaxgc' option.",
    )

    # Masking and output alignment options
    parser.add_argument(
        '--mask',
        default=False,
        action='store_true',
        help='Mask corrected positions in alignment with degenerate IUPAC codes.',
    )
    parser.add_argument(
        '--noappend',
        default=False,
        action='store_true',
        help="If set, do not append deRIP'd sequence to output alignment.",
    )

    # Output file options
    parser.add_argument(
        '-d',
        '--outDir',
        type=str,
        default=None,
        help="Directory for deRIP'd sequence files to be written to.",
    )
    parser.add_argument(
        '-o',
        '--outFasta',
        default=None,
        help='Write un-gapped RIP-corrected sequence to this file in fasta format. Default: deRIP_output.fa',
    )
    parser.add_argument(
        '--outAln',
        default=None,
        help='Optional: If set write alignment including deRIP corrected sequence to this file.',
    )
    parser.add_argument(
        '--outAlnFormat',
        default='fasta',
        choices=['fasta', 'nexus'],
        help='Optional: Write alignment including deRIP sequence to file of format X. Default: fasta',
    )
    parser.add_argument(
        '--label',
        default='deRIPseq',
        help="Use label as name for deRIP'd sequence in output files.",
    )
    # Logging options
    parser.add_argument(
        '--loglevel',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level.',
    )
    parser.add_argument(
        '--logfile',
        default=None,
        help='Log file path.',
    )

    # Add terminal UI support
    add_tui_argument(parser, option_strings=['--tui'])

    # Parse arguments
    args = parser.parse_args()

    return args


def main() -> None:
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

    Returns
    -------
    None
        Does not return any values, but writes output files and logs to the console.
    """
    # ---------- Setup ----------
    # Get command line arguments
    args = mainArgs()
    # Print full sys.argv[0] command line call
    print(f'Command line call: {colored.green(" ".join(sys.argv))}')

    # Check/create output directory
    outDir, logfile = dochecks(args.outDir, args.logfile)

    # Set up logging based on specified level
    init_logging(loglevel=args.loglevel, logfile=logfile)

    # Set output file paths
    if args.outFasta:
        outPathFasta = path.join(outDir, args.outFasta)
    else:
        outPathFasta = None

    if args.outAln:
        outPathAln = path.join(outDir, args.outAln)
    else:
        outPathAln = None

    # ---------- Alignment Processing ----------
    # Read in alignment file, check at least 2 sequences present and names are unique
    align = ao.loadAlign(args.inAln, args.format)

    # Report alignment summary
    ao.alignSummary(align)

    # ---------- Initialize Tracking Data Structures ----------
    # Initialize object to assemble deRIP'd sequence
    tracker = ao.initTracker(align)

    # Initialize object to track RIP observations and GC content by row
    RIPcounts = ao.initRIPCounter(align)

    # ---------- Initial Processing ----------
    # Set invariant or highly gapped positions in final sequence
    tracker = ao.fillConserved(align, tracker, args.maxGaps)

    # ---------- RIP Correction ----------
    # Detect and correct RIP mutations, optionally mask positions
    tracker, RIPcounts, maskedAlign, corrected_positions = ao.correctRIP(
        align,
        tracker,
        RIPcounts,
        maxSNPnoise=args.maxSNPnoise,
        minRIPlike=args.minRIPlike,
        reaminate=args.reaminate,
        mask=args.mask,
    )

    # Report RIP counts per sequence
    ao.summarizeRIP(RIPcounts)

    # ---------- Fill Remaining Positions ----------
    # Select reference sequence for filling remaining uncorrected positions
    if not args.fillindex:
        # Select least RIP'd sequence (or most GC-rich if no RIP or tied for min-RIP)
        refID = ao.setRefSeq(align, RIPcounts, getMinRIP=True, getMaxGC=args.fillmaxgc)
    else:
        # Check that the specified row index exists
        ao.checkrow(align, idx=args.fillindex)
        # Use the user-specified row index
        refID = args.fillindex

    # Fill remaining unset positions from reference sequence
    tracker = ao.fillRemainder(align, refID, tracker)

    # ---------- Output Results ----------
    # Report deRIP'd sequence to stdout
    logging.info(f'Final RIP corrected sequence: {args.label}')
    ao.writeDERIP2stdout(tracker, ID=args.label)

    # Write deRIP'd sequence to FASTA file if requested
    if outPathFasta:
        logging.info(f"Writing deRIP'd sequence to file: {outPathFasta}")
        ao.writeDERIP(tracker, outPathFasta, ID=args.label)

    # Write alignment file with deRIP'd sequence if requested
    if args.outAln:
        logging.info('Preparing output alignment.')

        # Log if deRIP'd sequence will be appended to alignment
        if not args.noappend:
            logging.info(
                f'Appending corrected sequence to alignment with ID: {args.label}'
            )

        # Select alignment version based on masking option
        if args.mask:
            logging.info('Masking alignment columns with detected mutations.')
            outputAlign = maskedAlign
        else:
            outputAlign = align

        # Write the alignment to file
        logging.info(f'Writing modified alignment to path: {outPathAln}')
        ao.writeAlign(
            tracker,
            outputAlign,
            outPathAln,
            ID=args.label,
            outAlnFormat=args.outAlnFormat,
            noappend=args.noappend,
        )


if __name__ == '__main__':
    main()
