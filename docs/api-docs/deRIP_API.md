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
        - write_alignment
        - write_consensus
        - write_stats
        - write_html_report
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
