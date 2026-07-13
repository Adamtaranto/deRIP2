#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
``derip2-spectra`` — trinucleotide-context mutation-spectrum analysis.

Takes a multiple-sequence DNA alignment, reconstructs the deRIP'd ancestral
consensus, and calls every substitution of each sequence against that ancestor to
build SigProfiler SBS-96 (pyrimidine-collapsed) and SBS-192 (strand-resolved)
mutation spectra. Writes the SigProfiler-compliant matrix files, native spectrum
plots, an event table and a homoplasy (recurrence) report.

This is the tree-free *baseline* method. It is a separate entry point from the
``derip2`` consensus-correction command; the two share the same alignment
machinery but answer different questions.
"""

import csv
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


def _write_events_tsv(result, out_path: str) -> None:
    """
    Write every counted substitution event to a TSV file.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra.
    out_path : str
        Destination path.

    Returns
    -------
    None
        The file is written as a side effect.
    """
    records = result.event_records()
    fields = [
        'sample',
        'row',
        'col',
        'ref',
        'alt',
        'five_prime',
        'three_prime',
        'sbs96',
        'sbs192',
    ]
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(records)


def _write_homoplasy_tsv(result, out_path: str, min_hits: int = 2) -> None:
    """
    Write the homoplasy (multi-hit column) table to a TSV file.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra.
    out_path : str
        Destination path.
    min_hits : int, optional
        Minimum independent hits per site to report (default: 2).

    Returns
    -------
    None
        The file is written as a side effect.
    """
    rows = result.homoplasy_table(min_hits=min_hits)
    fields = ['col', 'ref', 'alt', 'n_independent']
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Build SBS-96 and SBS-192 trinucleotide mutation spectra from a DNA '
    "alignment by calling substitutions against the deRIP'd ancestral consensus.",
)
@click.version_option(version=__version__, prog_name='derip2-spectra')
# Input / output
@click.option(
    '-i',
    '--input',
    required=True,
    type=str,
    help='Multiple sequence alignment (FASTA, optionally gzipped).',
)
@click.option(
    '-d',
    '--out-dir',
    type=str,
    default=None,
    help='Directory for spectrum output files.',
)
@click.option(
    '-p',
    '--prefix',
    default='deRIPspectra',
    show_default=True,
    help='Prefix for output files.',
)
@click.option(
    '--ancestor',
    type=str,
    default=None,
    help='Optional FASTA of a hypothetical ancestor to call against '
    'instead of the reconstructed deRIP consensus. Must be the same '
    'length as the alignment.',
)
# Spectrum options
@click.option(
    '--sbs',
    type=click.Choice(['96', '192', 'both']),
    default='both',
    show_default=True,
    help='Which SBS matrices/plots to produce.',
)
@click.option(
    '--partition-by',
    type=click.Choice(['none', 'row']),
    default='none',
    show_default=True,
    help='Split spectra into one pooled sample or one sample per sequence.',
)
@click.option(
    '--percentage',
    is_flag=True,
    default=False,
    show_default=True,
    help='Plot spectra as a percentage of each sample total.',
)
@click.option(
    '--min-hits',
    type=int,
    default=2,
    show_default=True,
    help='Minimum independent hits for a site in the homoplasy report.',
)
@click.option(
    '--no-plots',
    is_flag=True,
    default=False,
    show_default=True,
    help='Write matrices and tables only; skip figures.',
)
# deRIP parameters used to build the ancestor
@click.option(
    '-g',
    '--max-gaps',
    type=float,
    default=0.7,
    show_default=True,
    help='Maximum gap proportion in a column before it is gapped in the consensus.',
)
@click.option(
    '-a',
    '--reaminate',
    is_flag=True,
    default=False,
    show_default=True,
    help='Correct all deamination events regardless of RIP context '
    'when building the ancestor.',
)
@click.option(
    '--max-snp-noise',
    type=float,
    default=0.5,
    show_default=True,
    help='Maximum proportion of conflicting SNPs before a column is '
    'excluded from RIP assessment.',
)
@click.option(
    '--min-rip-like',
    type=float,
    default=0.1,
    show_default=True,
    help='Minimum proportion of RIP-context deamination for a column to be corrected.',
)
@click.option(
    '--fill-max-gc',
    is_flag=True,
    default=False,
    show_default=True,
    help='Fill uncorrected positions from the highest-GC sequence '
    "rather than the least-RIP'd one.",
)
@click.option(
    '--fill-index',
    type=int,
    default=None,
    help='Force the fill row by index (overrides --fill-max-gc).',
)
# Logging
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
    out_dir,
    prefix,
    ancestor,
    sbs,
    partition_by,
    percentage,
    min_hits,
    no_plots,
    max_gaps,
    reaminate,
    max_snp_noise,
    min_rip_like,
    fill_max_gc,
    fill_index,
    loglevel,
    logfile,
):
    """
    Build and write trinucleotide mutation spectra for an alignment.

    Reconstructs the deRIP'd ancestral consensus, calls every substitution of
    each sequence against it, and writes SBS-96/192 matrices, native spectrum
    plots, an event table and a homoplasy report.

    Parameters
    ----------
    input : str
        Path to the multiple sequence alignment.
    out_dir : str or None
        Output directory; the current directory is used when None.
    prefix : str
        Prefix for all output file names.
    ancestor : str or None
        Optional FASTA of a hypothetical ancestor to call against; when None the
        reconstructed deRIP consensus is used.
    sbs : {'96', '192', 'both'}
        Which SBS matrices and plots to produce.
    partition_by : {'none', 'row'}
        Whether to pool sequences into one sample or split per sequence.
    percentage : bool
        Plot spectra as percentages rather than counts.
    min_hits : int
        Minimum independent hits for a site in the homoplasy report.
    no_plots : bool
        If True, skip figure generation.
    max_gaps : float
        Maximum gap proportion in a column before it is gapped in the consensus.
    reaminate : bool
        Correct all deamination events regardless of RIP context.
    max_snp_noise : float
        Maximum proportion of conflicting SNPs before excluding a column.
    min_rip_like : float
        Minimum RIP-context deamination proportion for correction.
    fill_max_gc : bool
        Fill uncorrected positions from the highest-GC sequence.
    fill_index : int or None
        Force the fill row by index.
    loglevel : str
        Logging level.
    logfile : str or None
        Log file path.

    Returns
    -------
    None
        Writes output files and logs; returns nothing.
    """
    print(f'Command line call: {colored.green(" ".join(sys.argv))}\n')

    out_dir, logfile = dochecks(out_dir, logfile)
    init_logging(loglevel=loglevel, logfile=logfile)

    logger.info(f'Processing alignment file: \033[0m{input}')
    derip_obj = DeRIP(
        alignment_input=input,
        max_snp_noise=max_snp_noise,
        min_rip_like=min_rip_like,
        reaminate=reaminate,
        fill_index=fill_index,
        fill_max_gc=fill_max_gc,
        max_gaps=max_gaps,
    )
    logger.info(f'Loaded alignment with {len(derip_obj.alignment)} sequences')
    derip_obj.calculate_rip(label=prefix)

    # Resolve the ancestral reference: user-supplied FASTA or the deRIP consensus.
    ancestor_seq = None
    if ancestor is not None:
        ancestor_aln = ao.loadAlign(ancestor, alnFormat='fasta')
        ancestor_seq = str(ancestor_aln[0].seq)
        logger.info(f'Using supplied ancestor: \033[0m{ancestor}')

    result = derip_obj.calculate_spectra(
        partition_by=partition_by, ancestor=ancestor_seq
    )
    logger.info(
        f'Called {result.event_rows.size} substitution events '
        f'({result.n_indel_or_ambiguous} indel/ambiguous skipped, '
        f'{result.n_unassignable_context} without full context)'
    )

    kinds = ['96', '192'] if sbs == 'both' else [sbs]

    # Matrices.
    for kind in kinds:
        matrix_path = path.join(out_dir, f'{prefix}.SBS{kind}.txt')
        logger.info(f'Writing SBS-{kind} matrix to: \033[0m{matrix_path}')
        derip_obj.write_spectra_matrix(matrix_path, kind=kind)

    # Event and homoplasy tables.
    events_path = path.join(out_dir, f'{prefix}_events.tsv')
    homoplasy_path = path.join(out_dir, f'{prefix}_homoplasy.tsv')
    logger.info(f'Writing event table to: \033[0m{events_path}')
    _write_events_tsv(result, events_path)
    logger.info(f'Writing homoplasy report to: \033[0m{homoplasy_path}')
    _write_homoplasy_tsv(result, homoplasy_path, min_hits=min_hits)

    # Figures.
    if not no_plots:
        for kind in kinds:
            fig_path = path.join(out_dir, f'{prefix}_SBS{kind}.png')
            logger.info(f'Plotting SBS-{kind} spectrum to: \033[0m{fig_path}')
            derip_obj.plot_spectra(
                fig_path,
                kind=kind,
                title=f'{prefix} SBS-{kind}',
                percentage=percentage,
            )
        if '192' in kinds:
            asym_path = path.join(out_dir, f'{prefix}_strand_asymmetry.png')
            logger.info(f'Plotting strand asymmetry to: \033[0m{asym_path}')
            derip_obj.plot_spectra(
                asym_path, kind='strand', title=f'{prefix} strand asymmetry'
            )
        hom_fig = path.join(out_dir, f'{prefix}_homoplasy.png')
        logger.info(f'Plotting homoplasy to: \033[0m{hom_fig}')
        derip_obj.plot_spectra(
            hom_fig,
            kind='homoplasy',
            min_hits=min_hits,
            title=f'{prefix} recurrent sites',
        )

    logger.info('Mutation spectrum analysis complete.')


if __name__ == '__main__':
    main()
