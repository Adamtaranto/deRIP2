# DeRIP Class

::: derip2.derip.DeRIP
    options:
      members:
        - __init__
        - calculate_rip
        - calculate_cri
        - calculate_cri_for_all
        - calculate_dinucleotide_frequency
        - calculate_rsi
        - rip_summary
        - summarize_cri
        - summarize_stats
        - stats_summary
        - calculate_spectra
        - write_spectra_matrix
        - plot_spectra
        - calculate_flank_spectra
        - write_flank_spectra_matrix
        - write_flank_spectra_comparisons
        - plot_flank_spectra
        - plot_flank_conversion_heatmap
        - calculate_max_rip
        - get_max_rip_string
        - get_max_rip_positions
        - write_max_rip
        - write_alignment
        - write_consensus
        - write_stats
        - write_html_report
        - write_per_sequence_report
        - plot_alignment
        - plot_strand_bias
        - get_cri_values
        - get_rsi_values
        - get_gc_content
        - get_consensus_string
        - sort_by_cri
        - sort_by_rsi
        - filter_by_cri
        - filter_by_gc
        - keep_low_cri
        - keep_high_gc

# Strand bias statistics

::: derip2.stats.strand_bias
    options:
      members:
        - compute_rsi
        - RSIResult

# RIP column classification

::: derip2.aln_ops
    options:
      members:
        - ColumnClassification
        - classify_columns
        - classify_alignment
        - apply_classification

# Strand bias plotting

::: derip2.plotting.strandbias
    options:
      members:
        - plot_strand_bias

::: derip2.report
    options:
      members:
        - write_html_report

# Per-sequence reporting

::: derip2.plotting.persequence
    options:
      members:
        - per_sequence_strand_bias
        - sequence_row_strip
        - rip_completion_bar
        - gc_content_bar
        - resolve_cmap
        - text_color_on

::: derip2.persequence_report
    options:
      members:
        - write_per_sequence_report

# Gene annotation and RIP effect prediction

::: derip2.annotation
    options:
      members:
        - parse_gff3
        - Gene
        - Feature
        - EffectRecord
        - ungapped_to_column_map
        - predict_gene_effects
        - translate_cds
        - compute_effects_for_alignment
        - deripd_translations
        - write_snp_effects
        - build_annotation_spans
        - load_annotation_colors

# Mutation spectra statistics

::: derip2.stats.mutation_spectra
    options:
      members:
        - SpectraResult
        - compute_spectra
        - assemble_matrices
        - assemble_downstream

# Mutation spectra plotting

::: derip2.plotting.spectra
    options:
      members:
        - plot_sbs96
        - plot_sbs192
        - plot_downstream
        - strand_asymmetry
        - plot_strand_asymmetry
        - plot_homoplasy

# Spectra comparison

::: derip2.stats.spectra_compare
    options:
      members:
        - cosine_similarity
        - chi2_homogeneity
        - compare_spectra
        - compare_matrix_files
        - pairwise_compare

# Flank-context spectra statistics

::: derip2.stats.flank_spectra
    options:
      members:
        - FlankSpectraResult
        - compute_flank_spectra
        - compare_flank_spectra
        - compare_flank_spectra_pooled
        - differential_channels
        - write_flank_matrix
        - write_flank_comparisons

# Flank-context spectra plotting

::: derip2.plotting.flank_spectra
    options:
      members:
        - plot_flank_bihistograms
        - plot_flank_bihistograms_pooled
        - plot_flank_conversion_heatmap

# Maximum RIP sequences

::: derip2.maxrip
    options:
      members:
        - MaxRIPResult
        - compute_max_rip
        - write_max_rip_fasta
        - max_rip_multifasta

# Phylogenetic spectra (ancestral state reconstruction)

::: derip2.spectra.tree_asr
    options:
      members:
        - build_reconstruction
        - reconstruct
        - assign_clades
        - assign_groups
        - run_iqtree
        - find_iqtree
        - iqtree_version
        - TreeReconstruction

::: derip2.spectra.call_mutations
    options:
      members:
        - compute_spectra_from_tree

# Spectra channels and matrix IO

::: derip2.spectra.channels
    options:
      members:
        - sbs96_channel
        - sbs192_channel
        - downstream_channel
        - trinucleotide_context
        - downstream_context
        - fold_to_pyrimidine
        - revcomp_base

::: derip2.spectra.flank_channels
    options:
      members:
        - flank_channel_labels
        - flank_pair_labels

::: derip2.spectra.matrix_io
    options:
      members:
        - write_sbs_matrix
        - read_sbs_matrix
        - write_matrix_metadata

# Alignment QC for spectra

::: derip2.spectra.qc
    options:
      members:
        - ColumnProfile
        - profile_alignment
        - write_column_profile
        - write_qc_report
