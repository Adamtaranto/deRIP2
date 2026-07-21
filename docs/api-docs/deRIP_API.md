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
