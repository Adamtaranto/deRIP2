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
    # Phylogenetic events additionally carry the edge's parent/child node names.
    if result.event_child_names is not None:
        fields += ['parent', 'child']
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


# Column headers that mark a groups file's optional first row, to be skipped.
_GROUP_HEADER_KEYS = {'name', 'id', 'sequence', 'seq', 'taxon', 'tip'}


def _load_group_lookup(path):
    """
    Load a sequence-to-group mapping and return a sanitisation-tolerant lookup.

    The file is two whitespace- or tab-separated columns: a sequence name and its
    group label (e.g. a species). Blank lines and ``#`` comments are ignored, and
    an optional header row (first token one of ``name``/``id``/``sequence``/…) is
    skipped.

    Parameters
    ----------
    path : str
        Path to the mapping file.

    Returns
    -------
    callable
        ``lookup(name) -> str or None`` that resolves a sequence name to its
        group, matching either the original name or its IQ-TREE-sanitised form
        (so tree tip names resolve too).

    Raises
    ------
    ValueError
        If no usable name/group pairs are found.
    """
    from derip2.spectra.tree_asr import _sanitize_name

    by_original = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, label = parts[0], parts[1]
            if not by_original and name.lower() in _GROUP_HEADER_KEYS:
                continue  # skip a header row
            by_original[name] = label

    if not by_original:
        raise ValueError(f'No sequence/group pairs found in {path}')

    by_sanitized = {_sanitize_name(name): label for name, label in by_original.items()}

    def _lookup(name):
        if name in by_original:
            return by_original[name]
        return by_sanitized.get(_sanitize_name(name))

    return _lookup


def _run_phylo(
    derip_obj,
    out_dir,
    prefix,
    *,
    tree,
    iqtree_model,
    threads,
    rooting,
    outgroup,
    partition_by,
    groups,
    min_prob,
    root_sensitivity,
):
    """
    Run the phylogenetic path: QC, IQ-TREE reconstruction and branch traversal.

    Parameters
    ----------
    derip_obj : derip2.derip.DeRIP
        The alignment wrapper, already RIP-processed.
    out_dir : str
        Output directory for QC files and the run manifest.
    prefix : str
        Output file prefix.
    tree : str or None
        Fixed Newick tree, or None to infer one.
    iqtree_model : str
        IQ-TREE substitution model.
    threads : str
        Value for IQ-TREE ``-T`` (e.g. ``'AUTO'`` or an integer string).
    rooting : {'midpoint', 'outgroup', 'none'}
        Rooting strategy.
    outgroup : str or None
        Comma-separated outgroup tip name(s).
    partition_by : {'none', 'clade'}
        Sample partitioning; ``'clade'`` splits by root subtree. Ignored when
        ``groups`` is given.
    groups : str or None
        Path to a sequence-to-group mapping; branches whose descendants all share
        one group are attributed to it (overrides ``partition_by``).
    min_prob : float
        Minimum parent x child posterior for an event to be kept.
    root_sensitivity : bool
        Whether to report the direction-flip fraction under midpoint rooting.

    Returns
    -------
    derip2.stats.mutation_spectra.SpectraResult
        The phylogenetic spectra.
    """
    import json

    from derip2.spectra import qc
    from derip2.spectra.call_mutations import compute_spectra_from_tree
    from derip2.spectra.tree_asr import (
        assign_clades,
        assign_groups,
        build_reconstruction,
        orientation_flip_fraction,
        reconstruct,
    )

    if partition_by == 'row':
        raise click.UsageError(
            "--partition-by row is only valid for --method baseline; use 'clade'"
        )

    # QC profile.
    profile = qc.profile_alignment(derip_obj.alignment)
    qc.write_qc_report(
        derip_obj.alignment, profile, path.join(out_dir, f'{prefix}_qc_report.txt')
    )
    qc.write_column_profile(
        profile, path.join(out_dir, f'{prefix}_column_gap_profile.tsv')
    )

    outgroup_names = (
        [name.strip() for name in outgroup.split(',')] if outgroup else None
    )
    if outgroup_names and len(outgroup_names) == 1:
        outgroup_names = outgroup_names[0]

    work_prefix = path.join(out_dir, f'{prefix}_iqtree')
    logger.info('Running IQ-TREE ancestral reconstruction...')
    rec = reconstruct(
        derip_obj.alignment,
        work_prefix,
        model=iqtree_model,
        threads=threads,
        rooting=rooting,
        outgroup=outgroup_names,
        tree=tree,
    )

    # Sample assignment: user-defined groups take precedence over clade splitting.
    if groups:
        lookup = _load_group_lookup(groups)
        group_by_tip = {tip: (lookup(tip) or 'ungrouped') for tip in rec.tip_names}
        samples_by_child = assign_groups(rec, group_by_tip)
    elif partition_by == 'clade':
        samples_by_child = assign_clades(rec)
    else:
        samples_by_child = None
    result = compute_spectra_from_tree(
        rec, samples_by_child=samples_by_child, min_prob=min_prob
    )

    manifest = dict(rec.manifest)
    manifest['min_prob'] = min_prob
    manifest['partition_by'] = 'groups' if groups else partition_by

    # Root-sensitivity: how many edges flip direction under midpoint rooting.
    if root_sensitivity and rooting != 'midpoint':
        outputs_prefix = work_prefix
        mid = build_reconstruction(
            f'{outputs_prefix}.treefile',
            f'{outputs_prefix}.state',
            derip_obj.alignment,
            rooting='midpoint',
        )
        flip = orientation_flip_fraction(rec.edges, mid.edges)
        manifest['midpoint_direction_flip_fraction'] = flip
        logger.info('Root sensitivity: %.1f%% of edges flip under midpoint', flip * 100)

    manifest_path = path.join(out_dir, f'{prefix}_run_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, default=str)
    logger.info(f'Wrote run manifest to: \033[0m{manifest_path}')

    return result


def _write_outputs(result, out_dir, prefix, *, sbs, min_hits, percentage, no_plots):
    """
    Write matrices, tables and figures for a computed spectra result.

    Method-agnostic: works for both the baseline and phylogenetic results.

    Parameters
    ----------
    result : derip2.stats.mutation_spectra.SpectraResult
        The computed spectra.
    out_dir : str
        Output directory.
    prefix : str
        Output file prefix.
    sbs : {'96', '192', 'both'}
        Which matrices/plots to produce.
    min_hits : int
        Minimum independent hits for the homoplasy report/plot.
    percentage : bool
        Plot spectra as percentages.
    no_plots : bool
        Skip figures if True.

    Returns
    -------
    None
        Files are written as a side effect.
    """
    from derip2.plotting import spectra as spectra_plots
    from derip2.spectra import write_sbs_matrix

    kinds = ['96', '192'] if sbs == 'both' else [sbs]

    for kind in kinds:
        matrix_path = path.join(out_dir, f'{prefix}.SBS{kind}.txt')
        logger.info(f'Writing SBS-{kind} matrix to: \033[0m{matrix_path}')
        write_sbs_matrix(result, matrix_path, kind=kind)

    events_path = path.join(out_dir, f'{prefix}_events.tsv')
    homoplasy_path = path.join(out_dir, f'{prefix}_homoplasy.tsv')
    logger.info(f'Writing event table to: \033[0m{events_path}')
    _write_events_tsv(result, events_path)
    logger.info(f'Writing homoplasy report to: \033[0m{homoplasy_path}')
    _write_homoplasy_tsv(result, homoplasy_path, min_hits=min_hits)

    if no_plots:
        return
    for kind in kinds:
        fig_path = path.join(out_dir, f'{prefix}_SBS{kind}.png')
        logger.info(f'Plotting SBS-{kind} spectrum to: \033[0m{fig_path}')
        plotter = (
            spectra_plots.plot_sbs96 if kind == '96' else spectra_plots.plot_sbs192
        )
        plotter(result, fig_path, title=f'{prefix} SBS-{kind}', percentage=percentage)
    if '192' in kinds:
        asym_path = path.join(out_dir, f'{prefix}_strand_asymmetry.png')
        logger.info(f'Plotting strand asymmetry to: \033[0m{asym_path}')
        spectra_plots.plot_strand_asymmetry(
            result, asym_path, title=f'{prefix} strand asymmetry'
        )
    hom_fig = path.join(out_dir, f'{prefix}_homoplasy.png')
    logger.info(f'Plotting homoplasy to: \033[0m{hom_fig}')
    spectra_plots.plot_homoplasy(
        result, hom_fig, min_hits=min_hits, title=f'{prefix} recurrent sites'
    )


@click.command(
    context_settings={'help_option_names': ['-h', '--help']},
    help='Build SBS-96 and SBS-192 trinucleotide mutation spectra from a DNA '
    "alignment by calling substitutions against the deRIP'd ancestral consensus, "
    'or via IQ-TREE ancestral reconstruction (--method phylo).',
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
    type=click.Choice(['none', 'row', 'clade']),
    default='none',
    show_default=True,
    help='Split spectra into one pooled sample, one per sequence (baseline) or '
    'one per root clade (phylo).',
)
@click.option(
    '--groups',
    type=str,
    default=None,
    help='Path to a two-column (name, group) file mapping sequences to group '
    'labels (e.g. species). Reports one spectrum per group; works for both '
    'methods and tolerates IQ-TREE name reformatting. Overrides --partition-by.',
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
# Phylogenetic path (IQ-TREE ancestral reconstruction)
@click.option(
    '--method',
    type=click.Choice(['baseline', 'phylo']),
    default='baseline',
    show_default=True,
    help='Spectrum method: tree-free single-reference baseline, or phylogenetic '
    'branch-by-branch calling via IQ-TREE ancestral reconstruction.',
)
@click.option(
    '--tree',
    type=str,
    default=None,
    help='Fixed Newick tree for the phylo path; IQ-TREE reconstructs ancestral '
    'states on this topology instead of inferring a new tree.',
)
@click.option(
    '--iqtree-model',
    default='MFP',
    show_default=True,
    help='Substitution model passed to IQ-TREE (-m) for the phylo path.',
)
@click.option(
    '--threads',
    default='AUTO',
    show_default=True,
    help='IQ-TREE thread count (-T). AUTO benchmarks the best value; pass an '
    'integer to skip the benchmark (faster on small alignments).',
)
@click.option(
    '--rooting',
    type=click.Choice(['midpoint', 'outgroup', 'none']),
    default='midpoint',
    show_default=True,
    help='How to root the tree for the phylo path (sets substitution direction).',
)
@click.option(
    '--outgroup',
    default=None,
    help='Outgroup tip name(s) for --rooting outgroup; comma-separate a clade.',
)
@click.option(
    '--min-prob',
    type=float,
    default=0.0,
    show_default=True,
    help='Drop phylo events whose parent x child ancestral posterior is below '
    'this threshold.',
)
@click.option(
    '--root-sensitivity',
    is_flag=True,
    default=False,
    show_default=True,
    help='Also report the fraction of edges whose direction flips under midpoint '
    'rooting (phylo path).',
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
    groups,
    percentage,
    min_hits,
    no_plots,
    method,
    tree,
    iqtree_model,
    threads,
    rooting,
    outgroup,
    min_prob,
    root_sensitivity,
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
    partition_by : {'none', 'row', 'clade'}
        Whether to pool sequences into one sample, split per sequence (baseline)
        or per root clade (phylo).
    groups : str or None
        Path to a sequence-to-group mapping file; reports one spectrum per group
        (overrides ``partition_by``).
    percentage : bool
        Plot spectra as percentages rather than counts.
    min_hits : int
        Minimum independent hits for a site in the homoplasy report.
    no_plots : bool
        If True, skip figure generation.
    method : {'baseline', 'phylo'}
        Spectrum method: tree-free single-reference baseline, or phylogenetic
        branch-by-branch calling via IQ-TREE ancestral reconstruction.
    tree : str or None
        Fixed Newick tree for the phylo path, or None to infer one.
    iqtree_model : str
        Substitution model passed to IQ-TREE for the phylo path.
    threads : str
        IQ-TREE thread count (-T); ``'AUTO'`` or an integer string.
    rooting : {'midpoint', 'outgroup', 'none'}
        How to root the tree for the phylo path.
    outgroup : str or None
        Comma-separated outgroup tip name(s) for outgroup rooting.
    min_prob : float
        Drop phylo events below this parent x child posterior probability.
    root_sensitivity : bool
        Report the fraction of edges whose direction flips under midpoint rooting.
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

    # ---------- Compute the spectra by the chosen method ----------
    if method == 'baseline':
        if partition_by == 'clade':
            raise click.UsageError('--partition-by clade requires --method phylo')
        if groups:
            lookup = _load_group_lookup(groups)
            row_labels = [
                (lookup(record.id) or 'ungrouped') for record in derip_obj.alignment
            ]
            result = derip_obj.calculate_spectra(
                samples=row_labels, ancestor=ancestor_seq
            )
        else:
            result = derip_obj.calculate_spectra(
                partition_by=partition_by, ancestor=ancestor_seq
            )
    else:
        result = _run_phylo(
            derip_obj,
            out_dir,
            prefix,
            tree=tree,
            iqtree_model=iqtree_model,
            threads=threads,
            rooting=rooting,
            outgroup=outgroup,
            partition_by=partition_by,
            groups=groups,
            min_prob=min_prob,
            root_sensitivity=root_sensitivity,
        )

    logger.info(
        f'Called {result.event_rows.size} substitution events '
        f'({result.n_indel_or_ambiguous} indel/ambiguous skipped, '
        f'{result.n_unassignable_context} without full context)'
    )

    _write_outputs(
        result,
        out_dir,
        prefix,
        sbs=sbs,
        min_hits=min_hits,
        percentage=percentage,
        no_plots=no_plots,
    )
    logger.info('Mutation spectrum analysis complete.')


if __name__ == '__main__':
    main()
