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

logger = logging.getLogger(__name__)


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Predict ancestral sequence of fungal repeat elements by correcting for RIP-like mutations or cytosine deamination in multi-sequence DNA alignments. Optionally, mask mutated positions in alignment.',
)
@click.version_option(version=__version__, prog_name='derip2')
# Input options
@click.option(
    '-i', '--input', required=True, type=str, help='Multiple sequence alignment.'
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
@click.option(
    '--plot-format',
    type=click.Choice(['svg', 'png']),
    default='svg',
    show_default=True,
    help=(
        'File format for the --plot alignment visualization. "svg" is scalable '
        'vector output (the dense base grid of wide alignments is embedded as a '
        'raster and can blur at extreme zoom); "png" is a fully rasterised, '
        'high-resolution image that stays sharp at any zoom level.'
    ),
)
# Strand bias options
@click.option(
    '--plot-strand-bias',
    is_flag=True,
    default=False,
    show_default=True,
    help='Create a diverging stacked-bar chart of per-column RIP strand bias.',
)
@click.option(
    '--strand-bias-scale',
    type=click.Choice(['column', 'alignment', 'counts']),
    default='column',
    show_default=True,
    help=(
        'Bar height normalisation: each column to its own depth, to the number '
        'of sequences (so gappy columns are short), or raw counts.'
    ),
)
@click.option(
    '--strand-bias-xaxis',
    type=click.Choice(['none', 'logo', 'derip']),
    default='none',
    show_default=True,
    help="Draw a sequence logo or the deRIP'd consensus along the zero line.",
)
@click.option(
    '--strand-bias-columns',
    type=click.Choice(['rip', 'substrate', 'all']),
    default='all',
    show_default=True,
    help=(
        'Which positions are lettered along the zero line: RIP-like columns and '
        'their dinucleotide partners, untouched substrate columns and their '
        'partners, or every position. Every column is drawn as a bar regardless. '
        'Only has an effect with --strand-bias-xaxis logo or derip.'
    ),
)
@click.option(
    '--strand-bias-stack',
    type=click.Choice(['signal', 'product', 'all']),
    default='signal',
    show_default=True,
    help=(
        'Which bases each bar is made of: the RIP product and its unmutated '
        'substrate, the product alone, or every base with the remainder drawn '
        'translucent. Bars are never rescaled, so the missing height shows what '
        'was excluded.'
    ),
)
@click.option(
    '--rsi-ambiguous',
    type=click.Choice(['split', 'exclude', 'weight', 'both']),
    default='split',
    show_default=True,
    help=(
        'How to attribute a TA dinucleotide that could have arisen from RIP on '
        'either strand when calculating RSI.'
    ),
)
@click.option(
    '--sort-by-rsi',
    is_flag=True,
    default=False,
    show_default=True,
    help='Sort the output alignment from most forward- to most reverse-strand RIP.',
)
@click.option(
    '--stats-out',
    is_flag=True,
    default=False,
    show_default=True,
    help='Write the per-sequence statistics table to prefix_stats.tsv.',
)
@click.option(
    '--html-report',
    is_flag=True,
    default=False,
    show_default=True,
    help='Write a self-contained HTML report to prefix_report.html.',
)
@click.option(
    '--per-seq-report',
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        'Write an interactive per-sequence HTML report to '
        'prefix_per_sequence.html (one arrow-key-navigable panel per sequence).'
    ),
)
@click.option(
    '--max-report-seqs',
    type=int,
    default=None,
    show_default=True,
    help=(
        'Cap the number of sequence panels in the per-sequence report. When the '
        'alignment has more sequences, the strongest strand-bias sequences are '
        'kept. Unset renders every sequence.'
    ),
)
@click.option(
    '--spectra-ref-index',
    type=int,
    default=None,
    show_default=True,
    help=(
        'Alignment row index (0-based; negatives allowed) of a sequence to use '
        'as the reference for the per-sequence report mutation spectra, instead '
        'of the deRIP-corrected consensus. The chosen reference has an empty '
        '(self-comparison) spectrum. Unset compares against the deRIP consensus.'
    ),
)
# Gene annotation options
@click.option(
    '--gff',
    type=str,
    default=None,
    help=(
        'GFF3 gene model. Sequence ids must match alignment record ids. Enables '
        'a gene-annotation track on --plot, gene-effect panels in the '
        'per-sequence report, and a prefix_snp_effects.txt summary.'
    ),
)
@click.option(
    '--genetic-code',
    type=int,
    default=1,
    show_default=True,
    help='NCBI genetic code table for CDS translation and effect prediction.',
)
@click.option(
    '--annotation-colors',
    type=str,
    default=None,
    help=(
        'Two-column (type<TAB>hex) file overriding default annotation-track '
        'colours by feature type.'
    ),
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
    plot_format,
    plot_strand_bias,
    strand_bias_scale,
    strand_bias_xaxis,
    strand_bias_columns,
    strand_bias_stack,
    rsi_ambiguous,
    sort_by_rsi,
    stats_out,
    html_report,
    per_seq_report,
    max_report_seqs,
    spectra_ref_index,
    gff,
    genetic_code,
    annotation_colors,
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
    plot_format : str
        File format for the --plot alignment visualization. One of: 'svg' or 'png'.
        'svg' is scalable vector output; 'png' is a fully rasterised high-resolution
        image that stays sharp at any zoom level. Default: 'svg'.
    plot_strand_bias : bool
        If True, create a diverging stacked-bar chart of per-column RIP strand
        bias. Default: False.
    strand_bias_scale : str
        Bar height normalisation for the strand bias figure. One of: 'column',
        'alignment', or 'counts'. Default: 'column'.
    strand_bias_xaxis : str
        Decoration drawn along the zero line of the strand bias figure. One of:
        'none', 'logo', or 'derip'. Default: 'none'.
    strand_bias_columns : str
        Which positions are lettered along the zero line. One of: 'rip',
        'substrate', or 'all'. Every column is drawn as a bar regardless.
        Default: 'all'.
    strand_bias_stack : str
        Which bases each bar is made of. One of: 'signal', 'product', or 'all'.
        Default: 'signal'.
    rsi_ambiguous : str
        Policy for attributing TA products that either strand could explain. One
        of: 'split', 'exclude', 'weight', or 'both'. Default: 'split'.
    sort_by_rsi : bool
        If True, sort alignment rows from most forward-biased to most
        reverse-biased. Default: False.
    stats_out : bool
        If True, write the per-sequence statistics table as TSV. Default: False.
    html_report : bool
        If True, write a self-contained HTML strand bias report. Default: False.
    per_seq_report : bool
        If True, write an interactive per-sequence HTML report. Default: False.
    max_report_seqs : int or None
        Cap the number of sequence panels in the per-sequence report. If None,
        every sequence is rendered. Default: None.
    spectra_ref_index : int or None
        Alignment row index (0-based; negatives allowed) of a sequence to use as
        the reference for the per-sequence report mutation spectra, instead of the
        deRIP-corrected consensus. Default: None.
    gff : str or None
        Path to a GFF3 gene model. Enables the annotation track, gene-effect
        panels, and the SNP-effect summary. Default: None.
    genetic_code : int
        NCBI genetic code table for CDS translation. Default: 1.
    annotation_colors : str or None
        Path to a two-column annotation-track colour override file. Default: None.
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
    viz_path = path.join(out_dir, f'{prefix}_visualization.{plot_format}')
    strand_bias_path = path.join(out_dir, f'{prefix}_strand_bias.svg')
    stats_path = path.join(out_dir, f'{prefix}_stats.tsv')
    report_path = path.join(out_dir, f'{prefix}_report.html')
    per_seq_report_path = path.join(out_dir, f'{prefix}_per_sequence.html')
    snp_effects_path = path.join(out_dir, f'{prefix}_snp_effects.txt')
    flank_matrix_path = path.join(out_dir, f'{prefix}_rip_context_spectra.tsv')
    flank_compare_path = path.join(out_dir, f'{prefix}_rip_context_comparisons.tsv')

    # ---------- Create DeRIP object and process alignment ----------
    logger.info(f'Processing alignment file: \033[0m{input}')

    # Create DeRIP object with command line parameters
    derip_obj = DeRIP(
        alignment_input=input,
        max_snp_noise=max_snp_noise,
        min_rip_like=min_rip_like,
        reaminate=reaminate,
        fill_index=fill_index,
        fill_max_gc=fill_max_gc,
        max_gaps=max_gaps,
    )

    # Report alignment summary
    logger.info(f'Loaded alignment with {len(derip_obj.alignment)} sequences')
    ao.alignSummary(derip_obj.alignment)

    # Calculate RIP mutations and generate consensus
    logger.info('Processing alignment for RIP mutations...')
    derip_obj.calculate_rip(label=prefix)

    # Access corrected positions
    logger.info(
        f'\nDeRIP2 found {len(derip_obj.corrected_positions)} columns to be repaired.\n'
    )

    # Reordering rows changes what fill_index refers to, so re-run the analysis
    # against the new row order rather than reusing stale results.
    if sort_by_rsi:
        if fill_index is not None:
            logger.warning(
                '--fill-index refers to a row position, which --sort-by-rsi '
                'changes. The index is applied to the sorted alignment.'
            )
        logger.info('Sorting alignment by RIP strandedness imbalance')
        derip_obj.sort_by_rsi(inplace=True)
        derip_obj.fill_index = fill_index
        derip_obj.calculate_rip(label=prefix)

    # Print RIP summary
    logger.info(f'RIP summary by row:\n\033[0m{derip_obj.rip_summary()}\n')

    # Print the full per-sequence statistics table (CRI, PI, SI, GC, RSI)
    logger.info(
        f'Per-sequence statistics:\n\033[0m'
        f'{derip_obj.stats_summary(ambiguous=rsi_ambiguous)}\n'
    )

    pooled = derip_obj.rsi_result.pooled()
    logger.info(
        f'Pooled RIP strandedness imbalance: \033[0mRSI={pooled["RSI"]:+.3f} '
        f'(p_fwd={pooled["p_fwd"]:.3f}, p_rev={pooled["p_rev"]:.3f}, '
        f'p={pooled["pvalue"]:.3g}, {pooled["n_ambiguous"]} ambiguous TpA)'
    )

    # Print colourized alignment + consensus
    # If alignment dims are less than 100 columns x 50 rows
    if (
        len(derip_obj.alignment) < 50
        and derip_obj.alignment.get_alignment_length() < 100
    ):
        logger.info(f'Corrected alignment:\n\033[0m{derip_obj}\n')
    else:
        logger.debug(f'Corrected alignment:\n\033[0m{derip_obj}\n')
    # ---------- Output Results ----------
    # Report deRIP'd sequence to stdout
    logger.info(f'Final RIP corrected sequence: \033[0m{derip_obj.colored_consensus}')

    # Write deRIP'd sequence to FASTA file
    logger.info(f"Writing deRIP'd sequence to file: \033[0m{out_path_fasta}")
    derip_obj.write_consensus(out_path_fasta, consensus_id=prefix)

    # Write alignment file with deRIP'd sequence
    logger.info('Preparing output alignment.')

    # Log if deRIP'd sequence will be appended to alignment
    if not no_append:
        logger.info(
            f'Appending corrected sequence to alignment with ID: \033[0m{prefix}'
        )

    # Write the alignment to file
    logger.info(f'Writing alignment to path: \033[0m{out_path_aln}')
    derip_obj.write_alignment(
        output_file=out_path_aln,
        append_consensus=not no_append,
        mask_rip=mask,
        consensus_id=prefix,
        format='fasta',
    )

    # ---------- Gene annotation (GFF3) ----------
    # Parsed once and reused: an annotation track for --plot, gene effects for
    # the per-sequence report, and the SNP-effect summary written here.
    cds_tracks = None
    if gff:
        import numpy as np

        from derip2.annotation import (
            build_cds_tracks,
            compute_effects_for_alignment,
            deripd_translations,
            load_annotation_colors,
            parse_gff3,
            warn_unmatched_seqids,
            write_snp_effects,
        )

        logger.info(f'Reading gene annotation from: \033[0m{gff}')
        genes_by_seqid = parse_gff3(gff)
        warn_unmatched_seqids(genes_by_seqid, [r.id for r in derip_obj.alignment])

        colors = (
            load_annotation_colors(annotation_colors) if annotation_colors else None
        )
        row_lookup = {
            rec.id: derip_obj.column_classes.arr[i]
            for i, rec in enumerate(derip_obj.alignment)
        }
        # Rich CDS tracks for --plot: stop codons are read off the deRIP'd
        # consensus, so the track flags stops in the corrected reading frame.
        consensus_row = np.frombuffer(
            str(derip_obj.gapped_consensus.seq).upper().encode('ascii'), dtype='S1'
        )
        cds_tracks = build_cds_tracks(
            genes_by_seqid,
            row_lookup,
            consensus_row,
            genetic_code=genetic_code,
            colors=colors,
        )

        effects_by_seq = compute_effects_for_alignment(
            derip_obj, genes_by_seqid, genetic_code=genetic_code
        )
        deripd_aa = deripd_translations(
            derip_obj, genes_by_seqid, genetic_code=genetic_code
        )
        logger.info(f'Writing SNP-effect summary to: \033[0m{snp_effects_path}')
        write_snp_effects(snp_effects_path, effects_by_seq, deripd_aa)

    # Create visualization highlighting RIP/deamination events if requested
    if plot:
        logger.info(
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
            cds_tracks=cds_tracks,
        )

        if viz_result:
            logger.info(f'RIP visualization created at: \033[0m{viz_path}')
        else:
            logger.warning('Failed to create RIP visualization')

    # ---------- Strand bias outputs ----------
    strand_bias_opts = {
        'scale': strand_bias_scale,
        'xaxis': strand_bias_xaxis,
        'columns': strand_bias_columns,
        'stack': strand_bias_stack,
    }

    if plot_strand_bias:
        logger.info(f'Creating strand bias figure at: \033[0m{strand_bias_path}')
        try:
            derip_obj.plot_strand_bias(
                output_file=strand_bias_path,
                title=f'RIP strand bias: {prefix}',
                **strand_bias_opts,
            )
        except ValueError as error:
            logger.warning(f'Could not draw strand bias figure: {error}')

    if stats_out:
        logger.info(f'Writing statistics table to: \033[0m{stats_path}')
        derip_obj.write_stats(stats_path, ambiguous=rsi_ambiguous)

    if html_report:
        logger.info(f'Writing HTML report to: \033[0m{report_path}')
        derip_obj.write_html_report(
            report_path,
            title=f'deRIP2 strand bias: {prefix}',
            ambiguous=rsi_ambiguous,
            **strand_bias_opts,
        )

    if per_seq_report:
        if spectra_ref_index is not None:
            n_aln = len(derip_obj.alignment)
            if not -n_aln <= spectra_ref_index < n_aln:
                raise click.BadParameter(
                    f'{spectra_ref_index} is out of range for {n_aln} sequences '
                    f'(valid: {-n_aln}..{n_aln - 1}).',
                    param_hint='--spectra-ref-index',
                )
        logger.info(
            f'Writing per-sequence HTML report to: \033[0m{per_seq_report_path}'
        )
        derip_obj.write_per_sequence_report(
            per_seq_report_path,
            title=f'deRIP2 per-sequence: {prefix}',
            ambiguous=rsi_ambiguous,
            max_seqs=max_report_seqs,
            gff=gff,
            genetic_code=genetic_code,
            spectra_ref_index=spectra_ref_index,
        )
        # Companion tidy TSVs for the flank-context spectra of RIP-like sites:
        # the 16-channel counts and the per-sequence substrate-vs-product tests.
        logger.info(f'Writing RIP-context spectra to: \033[0m{flank_matrix_path}')
        derip_obj.write_flank_spectra_matrix(flank_matrix_path)
        derip_obj.write_flank_spectra_comparisons(flank_compare_path)


if __name__ == '__main__':
    main()
