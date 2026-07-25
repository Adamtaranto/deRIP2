<!--
This page is generated from docs/tutorial.ipynb (the editable source).
Regenerate it after editing the notebook with:
    jupyter nbconvert --to markdown docs/tutorial.ipynb
Do not edit tutorial.md by hand.
-->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adamtaranto/deRIP2/blob/main/docs/tutorial.ipynb)

!!! tip "Run this tutorial interactively"
    This is a static rendering of the tutorial notebook. Click the **Open In
    Colab** badge above to run it interactively in Google Colab, or open
    `docs/tutorial.ipynb` locally with Jupyter.

## deRIP2 Tutorial: Using the DeRIP Class Directly

This tutorial demonstrates how to use the `DeRIP` class from the `derip2` package directly in Python, without using the command-line interface. This gives you more programmatic control and flexibility when working with RIP analysis and correction.

## 1. Introduction

deRIP2 is a tool for detecting and correcting RIP (Repeat-Induced Point) mutations in fungal DNA alignments. RIP is a genome defense mechanism that introduces C→T mutations (and the complementary G→A on the opposite strand) in specific sequence contexts.

Key features of deRIP2:
- Predicts ancestral fungal transposon sequences by correcting for RIP-like mutations
- Masks RIP or deamination events as ambiguous bases
- Provides tools for analyzing RIP patterns and dinucleotide frequencies
- Calculates Composite RIP Index (CRI) and other metrics
- Offers visualization of alignments with RIP annotations

## 2. Installation and Setup

If you haven't installed deRIP2 yet, you can do so via pip:


```python
# Uncomment to install
# !pip install derip2

# For the latest development version
# !pip install git+https://github.com/Adamtaranto/deRIP2.git
```


```python
%matplotlib inline
import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import Image, display

# Import the DeRIP class
from derip2.derip import DeRIP


# Create output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
```

## 3. Loading and Examining an Alignment

First, let's load a multiple sequence alignment file. For this tutorial, we'll use the example file included with deRIP2.


```python
# If running this notebook on Google Colab, download the example alignment file
if 'google.colab' in str(get_ipython()):
    !wget -O mintest.fa https://raw.githubusercontent.com/Adamtaranto/deRIP2/main/tests/data/mintest.fa
    # Set the path to the alignment file
    alignment_file = 'mintest.fa'
else:
    # Set the path to the alignment file
    alignment_file = '../tests/data/mintest.fa'
```


```python
# Create a DeRIP object by loading the alignment
derip_obj = DeRIP(
    alignment_file,
    max_snp_noise=0.2,  # Maximum proportion of conflicting SNPs permitted
    min_rip_like=0.5,  # Minimum proportion of deamination events in RIP context
    max_gaps=0.7,  # Maximum proportion of gaps in a column)
    reaminate=False,  # Don't correct all deamination events
)
# Print basic information about the alignment
print(derip_obj)
```

## 4. Basic RIP Analysis and Correction

Now let's perform RIP detection and correction on the alignment. This is the core functionality of deRIP2.


```python
# Perform RIP correction
derip_obj.calculate_rip(label='deRIP_mintest')

# Access corrected positions
print(f'\nDeRIP2 found {len(derip_obj.corrected_positions)} columns to repair.')

# Print a summary of RIP mutations
rip_summary = derip_obj.rip_summary()
print('\nRIP Mutation Summary:')
print(rip_summary)

# Print colourized alignment + consensus
print('\nPrint function now returns colourized alignment + consensus:')
print(f'{derip_obj}')
# Target bases are bolded, substrate dinucleotides are blue, product dinucleotides are red
# Corrected positions in the consensus are highlighted in green
```

### Examining the Corrected Consensus Sequence

After running `calculate_rip()`, we can examine the corrected consensus sequence and see how it compares to the original sequences.


```python
# Get the corrected consensus sequence
consensus_seq = derip_obj.get_consensus_string()
print(f'Consensus sequence length: {len(consensus_seq)} bp')
print(f'First 100 bp: {consensus_seq[:100]}')

# Write the consensus sequence to a file
consensus_file = os.path.join(output_dir, 'consensus.fasta')

print(f'\nWriting consensus sequence to: {consensus_file}')

derip_obj.write_consensus(consensus_file, consensus_id='deRIP_mintest')
```


```python
# Write the alignment with the consensus sequence appended
out_alignment_file = os.path.join(output_dir, 'alignment_with_consensus.fasta')

print(f'Writing alignment with consensus to: {out_alignment_file}')

derip_obj.write_alignment(
    out_alignment_file,
    append_consensus=True,  # Append the consensus sequence to the alignment
    mask_rip=True,  # Mask RIP positions in the output alignment
    consensus_id='deRIP_mintest',
)
```

In the above example we used `mask_rip=True` to mask RIP events as ambiguous bases in the output alignment. If you want to keep the original sequences intact and only append the consensus sequence, you can set `mask_rip=False`.

You can preview the masked alignment using the `print_alignment()` method.b


```python
# Print masked alignment
print(f'Mutation masked alignment:\n{derip_obj.colored_masked_alignment}')
# Note: Consensus is not part of the masked alignment
print(f'{derip_obj.colored_consensus} {"deRIP_mintest"}\n')
```

## 5. Working with Composite RIP Index (CRI)

The Composite RIP Index (CRI) is a metric for measuring the extent of RIP mutations in a sequence. It combines two indices:

- Product Index (PI) = TpA / ApT
- Substrate Index (SI) = (CpA + TpG) / (ApC + GpT)
- CRI = PI - SI

A high CRI value indicates strong RIP activity. Let's analyze CRI values for our sequences.


```python
# Calculate CRI for all sequences in the alignment
derip_obj.calculate_cri_for_all()

# Get a summary table of CRI values
cri_summary = derip_obj.summarize_cri()
print(f'\nCRI Summary Table:\n{cri_summary}')
```

### Sorting and Filtering by CRI

We can sort sequences by their CRI values using `sort_by_cri()`. This can help identify sequences with the highest or lowest RIP activity.

By default `sort_by_cri()` returns a new `Bio.Align.Bio.MultipleSeqAlignment` object. If you want to modify the original alignment in place, you can set `inplace=True`.


```python
# Sort alignment by CRI values (descending order)
sorted_alignment = derip_obj.sort_by_cri(descending=True)

# Print sequences in order of descending CRI
print('Sequences sorted by CRI (lowest to highest):')
for i, record in enumerate(sorted_alignment):
    print(f'{i + 1}. {record.id}: CRI={record.annotations["CRI"]:.4f}')
```


```python
# Filter alignment to keep only sequences with CRI above a threshold
min_cri_threshold = -1.6

# Return a new alignment object with sequences that meet the criteria
# Set inplace=True to filter the original alignment object
filtered_alignment = derip_obj.filter_by_cri(min_cri=min_cri_threshold, inplace=False)

# Print the number of sequences that remain after filtering
print(
    f'After filtering (CRI >= {min_cri_threshold}): {len(filtered_alignment)} sequences remain'
)

# Print remaining records in the filtered alignment
for record in filtered_alignment:
    print(f'{record.id}: CRI={record.annotations["CRI"]:.4f}')
```

Instead of setting a threshold, you can also specify the number of sequences to keep using the `keep_low_cri()` method. This will keep the specified number of sequences with the lowest CRI values.


```python
# Keep only the top 3 sequences with the lowest CRI values
# By default, keep_low_cri() will return a new alignment object
# Set inplace=True to filter the original alignment object
three_lowest_cri_align = derip_obj.keep_low_cri(n=3, inplace=False)

# Print the number of sequences that remain after filtering
print(
    f'After keeping the 3 sequences with the lowest CRI values: {len(three_lowest_cri_align)} sequences remain'
)

# Print remaining records in the filtered alignment
print(three_lowest_cri_align)
```

## 6. GC Content Analysis

RIP mutations typically reduce GC content by converting C to T. Let's analyze the GC content distribution in our sequences:


```python
# Calculate GC content for all sequences
gc_values = derip_obj.get_gc_content()

# Print summary of GC content statistics
gc_content_values = [item['GC_content'] for item in gc_values]
print('GC Content Summary:')
print(f'  Min: {min(gc_content_values):.4f} ({min(gc_content_values) * 100:.2f}%)')
print(f'  Max: {max(gc_content_values):.4f} ({max(gc_content_values) * 100:.2f}%)')
print(
    f'  Mean: {sum(gc_content_values) / len(gc_content_values):.4f} ({sum(gc_content_values) / len(gc_content_values) * 100:.2f}%)'
)

# Create DataFrame from gc_values
gc_df = pd.DataFrame(gc_values)

# Create histogram using seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data=gc_df, x='GC_content', kde=True, bins=10)
plt.title('Distribution of GC Content Across Sequences')
plt.xlabel('GC Content')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gc_content_histogram.png'), dpi=300)
plt.show()
```

### Filtering by GC Content

We can filter sequences based on their GC content:


```python
# Filter alignment to keep only sequences with GC content above a threshold
min_gc_threshold = 0.4
filtered_by_gc = derip_obj.filter_by_gc(min_gc=min_gc_threshold, inplace=False)

print(
    f'After filtering (GC >= {min_gc_threshold}): {len(filtered_by_gc)} sequences remain'
)
for record in filtered_by_gc:
    print(f'{record.id}: GC={record.annotations["GC_content"]:.4f}')
```

We can retain just the n records with the highest GC using the `keep_high_gc()` method.


```python
# Keep only the top 3 sequences with the highest GC content
top_gc_align = derip_obj.keep_high_gc(n=3, inplace=False)

print(
    f'After keeping the 3 sequences with the highest GC content: {len(top_gc_align)} sequences remain'
)

print(top_gc_align)
```

## 7. Visualizing Alignments

deRIP2 offers visualization capabilities to see the alignment with RIP sites highlighted. This is particularly useful for understanding where RIP corrections have been made.


```python
# Generate a visualization of the alignment with RIP markup
viz_file = os.path.join(output_dir, 'alignment_visualization.png')
derip_obj.plot_alignment(
    output_file=viz_file,
    dpi=300,
    title='Alignment with RIP Mutations Highlighted',
    width=20,
    height=15,
    show_rip='both',  # Show both substrate and product sites
    show_chars=True,  # Display sequence characters
    flag_corrected=True,  # Highlight corrected positions
)

# Display the image
display(Image(viz_file))
```

Let's create another visualization focusing only on RIP product sites (TpA dinucleotides resulting from RIP):


```python
# Generate a visualization focusing on RIP products
product_viz_file = os.path.join(output_dir, 'rip_products_visualization.png')
derip_obj.plot_alignment(
    output_file=product_viz_file,
    dpi=300,
    title='Alignment with RIP Product Sites Highlighted',
    show_rip='product',  # Show only product sites
    show_chars=True,
    flag_corrected=True,
)

# Display the image
display(Image(product_viz_file))
```

## 8. Re-running with Different Parameters

Let's demonstrate how to re-run the analysis with different parameters, such as enabling the `reaminate` option to correct all deamination events regardless of RIP context:


```python
# Create a new DeRIP instance with reaminate=True
derip_reaminate = DeRIP(
    alignment_file,
    max_snp_noise=0.2,
    min_rip_like=0.5,
    max_gaps=0.7,
    reaminate=True,  # Correct all deamination events
)

# Perform RIP correction with reamination
derip_reaminate.calculate_rip(label='deRIP_reaminate')

# Write the reaminated consensus sequence
reaminated_consensus_file = os.path.join(output_dir, 'reaminated_consensus.fasta')
derip_reaminate.write_consensus(
    reaminated_consensus_file, consensus_id='deRIP_reaminate'
)

# Create visualization
reaminate_viz_file = os.path.join(output_dir, 'reaminated_visualization.png')
derip_reaminate.plot_alignment(
    output_file=reaminate_viz_file,
    dpi=300,
    title='Alignment with All Deamination Events Corrected',
    show_rip='product',
    show_chars=True,
    highlight_corrected=True,
    flag_corrected=True,
)

# Display the image
display(Image(reaminate_viz_file))
```

## 9. Comparing Original and Reaminated Consensus Sequences

Let's compare the consensus sequences from both approaches to see how the `reaminate` option affects the results:


```python
# Get both consensus sequences
original_consensus = derip_obj.get_consensus_string()
reaminated_consensus = derip_reaminate.get_consensus_string()

# Find differences between the sequences
differences = []
for i, (orig, ream) in enumerate(
    zip(original_consensus, reaminated_consensus, strict=False)
):
    if orig != ream:
        differences.append((i, orig, ream))

# Print comparison
print(f'Total differences between consensus sequences: {len(differences)}')
print('\nFirst 10 differences (position, original, reaminated):')
for _i, (pos, orig, ream) in enumerate(differences[:10]):
    print(f'Position {pos + 1}: {orig} → {ream}')
```

## 10. Working with a real transposon alignment

In this section we will process a large alignment containing 396 copies of the DNA transposon *Sahana* from *Leptosphaeria maculans*.

Most copies are heavily RIP'd and some are fragmented. This aligment has been pre-curated with [TEtrimmer](https://github.com/Adamtaranto/TEtrimmer) to remove poorly aligned regions that might interfere with RIP detection.

### Loading the alignment



```python
# Fetch the gzipped alignment file from the deRIP2 GitHub repository if running on Google Colab
if 'google.colab' in str(get_ipython()):
    !wget -O sahana.fasta.gz https://raw.githubusercontent.com/Adamtaranto/deRIP2/main/tests/data/sahana.fasta.gz
    # Set the path to the gzipped alignment file
    gzipped_alignment_file = 'sahana.fasta.gz'
else:
    # Set the path to the gzipped alignment file
    gzipped_alignment_file = '../tests/data/sahana.fasta.gz'
```


```python
# load the gzipped file
import os
from derip2.derip import DeRIP

# Load the alignment from gzipped file
sahana = DeRIP(gzipped_alignment_file)

# Inspect alignment stats
print(
    f'Alignment dimensions: {len(sahana.alignment)} sequences × {sahana.alignment.get_alignment_length()} columns'
)
print(
    f'\nSequence length range: {min([len(str(rec.seq).replace("-", "")) for rec in sahana.alignment])} - {max([len(str(rec.seq).replace("-", "")) for rec in sahana.alignment])} bp'
)

# Preview alignment
# print(f"\n{sahana.alignment}")
```

### Inspecting sequence properties

Next we will calculate the Composite RIP Index (CRI) and GC scores and inspect their distributions.


```python
# Calc CRI
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Calculate CRI values for all sequences
sahana.calculate_cri_for_all()

# Get CRI list
cri_values = sahana.get_cri_values()
cri_df = pd.DataFrame(cri_values)

# Plot CRI distribution
plt.figure(figsize=(10, 6))
sns.histplot(cri_df['CRI'], kde=True, bins=30)
plt.title('Distribution of CRI Values in Sahana Transposon Copies')
plt.xlabel('Composite RIP Index (CRI)')
plt.ylabel('Frequency')
plt.axvline(x=1, color='red', linestyle='--', label='CRI = 1.0')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Calculate GC content
gc_values = sahana.get_gc_content()
gc_df = pd.DataFrame(gc_values)

# Plot GC content distribution
plt.figure(figsize=(10, 6))
sns.histplot(gc_df['GC_content'], kde=True, bins=30)
plt.title('Distribution of GC Content in Sahana Transposon Copies')
plt.xlabel('GC Content')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.show()

# Summary statistics
print(
    f'CRI range: {cri_df["CRI"].min():.4f} - {cri_df["CRI"].max():.4f}, Mean: {cri_df["CRI"].mean():.4f}'
)
print(
    f'GC content range: {gc_df["GC_content"].min():.4f} - {gc_df["GC_content"].max():.4f}, Mean: {gc_df["GC_content"].mean():.4f}'
)
```

### Filtering the alignment

Generally it is best to include all available sequences in your DeRIP analysis to maximise discovery of un-mutated substrate motifs that can be used for sequence correction.

However, large alignments can take a long time to process.

Consider the following methods to reduce your alignment size:

- Filter out sequences with high CRI scores or low GC proportions
- Remove duplicate sequences
- Remove partial sequences

#### Removing partial sequences

First we will remove partial sequences that are less than 90% the length of the consensus sequence.

We can directly edit the `sahana.aligntment` attribute to remove these sequences.


```python
# Filter alignment rows that are < 90% of the alignment length (not including gaps)
from Bio.Align import MultipleSeqAlignment

# Calculate ungapped lengths for each sequence
ungapped_lengths = [len(str(rec.seq).replace('-', '')) for rec in sahana.alignment]
max_ungapped_length = max(ungapped_lengths)

# Set threshold at 90% of the maximum ungapped length
length_threshold = max_ungapped_length * 0.9
print(f'Maximum ungapped length: {max_ungapped_length} bp')
print(f'Length threshold (90%): {length_threshold:.1f} bp')

# Filter sequences that meet the threshold
filtered_records = []
for rec in sahana.alignment:
    ungapped_length = len(str(rec.seq).replace('-', ''))
    if ungapped_length >= length_threshold:
        filtered_records.append(rec)

# Create new MultipleSeqAlignment with the filtered sequences
filtered_alignment = MultipleSeqAlignment(filtered_records)

# Update sahana's alignment with the filtered one
sahana.alignment = filtered_alignment

print(f'Original alignment: {len(ungapped_lengths)} sequences')
print(f'Filtered alignment: {len(filtered_alignment)} sequences')
print(f'Removed: {len(ungapped_lengths) - len(filtered_alignment)} sequences')

# Preview the filtered alignment
# print(sahana)
```

#### Keep only the best 50 sequences by CRI

Low CRI scores are indicative of fewer RIP mutations. We will keep only the best 50 sequences by CRI score.

We can use the `keep_low_cri()` method with `inplace=True` to filter the alignment attribute.


```python
# Filter for lowest 50 CRI scores
filtered_sahana = sahana.keep_low_cri(n=50, inplace=True)

# Check new alignment dimensions
print(
    f'Filtered alignment dimensions: {len(sahana.alignment)} sequences × {sahana.alignment.get_alignment_length()} columns'
)

# Sort low to high CRI
sahana.sort_by_cri(descending=False, inplace=True)
print('\nAlignment sorted by CRI (lowest to highest)')
print('\nFirst 3 sequence IDs with their CRI values:')
for _i, rec in enumerate(sahana.alignment[:3]):
    print(f'{rec.id}: CRI = {rec.annotations["CRI"]:.4f}')

# Example on how to keep_high_gc()
# print("\nExample of keeping sequences with highest GC content:")
# print(f"Before filtering: {len(sahana.alignment)} sequences")
# This line would be executed if you wanted to keep only high GC sequences
# high_gc_alignment = sahana.keep_high_gc(n=50, inplace=False)
# print(f"After keeping top 50 high GC sequences: {len(high_gc_alignment)} sequences")
```

### Calculating RIP motifs and consensus

Now we will run `calculate_rip()` to identify RIP substrate and product motifs, and to calculate the deRIP'd consensus.


```python
# Run deRIP
sahana.calculate_rip(label='sahana_deRIP')

# Information about reference sequence used for filling
print(f'Reference sequence used for filling: index {sahana.fill_index}')

# Print summary of detected RIP mutations
print(sahana.rip_summary())
```

#### Inspect calculated RIP features

Lets inspect the object. Several new attributes are now available.


```python
# Print colored alignment (first 5 sequences only, to avoid overwhelming output)
print('Colored alignment view (first 5 sequences):')
colored_lines = sahana.colored_alignment.split('\n')
for line in colored_lines[:5]:
    print(line)
print('...')

# Normal and colored consensus
print('\nConsensus sequence (first 100 bp):')
print(sahana.get_consensus_string()[:100] + '...')

print('\nColored consensus sequence:')
print(sahana.colored_consensus)

# Report on remaining variable positions that were filled from reference
print(f'\nTotal positions corrected during deRIP: {len(sahana.corrected_positions)}')
```

### Writing the results to file

We can write the ungapped consensus, alignment +/- consensus, or masked alignment +/- consensus to fasta file.


```python
# Create an output directory if it doesn't exist
output_dir = 'sahana_output'
os.makedirs(output_dir, exist_ok=True)

# Write the ungapped consensus sequence
consensus_file = os.path.join(output_dir, 'sahana_derip_consensus.fasta')
sahana.write_consensus(consensus_file, consensus_id='sahana_deRIP')

# Write the alignment with consensus appended
alignment_file = os.path.join(output_dir, 'sahana_alignment_with_consensus.fasta')
sahana.write_alignment(
    output_file=alignment_file,
    append_consensus=True,
    mask_rip=False,  # Original sequences, not masked
    consensus_id='sahana_deRIP',
)

# Write the masked alignment with consensus appended
masked_alignment_file = os.path.join(
    output_dir, 'sahana_masked_alignment_with_consensus.fasta'
)
sahana.write_alignment(
    output_file=masked_alignment_file,
    append_consensus=True,
    mask_rip=True,  # Masked sequences
    consensus_id='sahana_deRIP',
)

print(f'Files written to {output_dir}:')
print(f'  - {os.path.basename(consensus_file)}')
print(f'  - {os.path.basename(alignment_file)}')
print(f'  - {os.path.basename(masked_alignment_file)}')
```

## Plotting processed alignment


```python
# Create a plot of the alignment
plot_file = os.path.join(output_dir, 'sahana_alignment_visualization.png')

sahana.plot_alignment(
    output_file=plot_file,
    dpi=300,
    title='Sahana Transposon Alignment with RIP Corrections',
    width=20,
    height=12,
    show_chars=False,  # Off for large alignments
    draw_boxes=False,  # Off for large alignments
    show_rip='product',  # Show RIP product only
    highlight_corrected=True,  # Highlight corrected positions
    flag_corrected=False,  # Off for large alignments
)

print(f'Plot saved to {plot_file}')

# Note: Variables that can customize the plot:
# - show_rip: Options are 'product', 'substrate', or 'both'
# - highlight_corrected: Whether to highlight corrected positions in consensus
# - flag_corrected: Whether to mark corrected positions with asterisks

# Display the generated plot
display(Image(plot_file))
```

## 11. Per-sequence strand-imbalance (RSI) statistics

A single round of meiotic RIP acts on only **one strand** of a duplex, so a sequence RIP'd in a single burst shows a **strand imbalance**: far more `CpA→TpA` conversions on one strand than the other. deRIP2 quantifies this per sequence as the **RIP Strandedness Imbalance (RSI)**, which ranges from `+1` (all forward-strand RIP), through `0` (balanced, or never RIP'd), to `−1` (all reverse-strand RIP).

The `summarize_stats()` / `stats_summary()` methods return a per-sequence table combining the RIP event counts, CRI, GC content, and the RSI components `p_fwd` and `p_rev`. `sort_by_rsi()` ranks the alignment by strandedness. See the [RIP strand bias tutorial](tutorials/rip-strand-bias.md) for how to read and interpret these columns (including why `RSI = 0` is ambiguous without `p_fwd`/`p_rev`).


```python
# Full per-sequence statistics table: RIP counts, CRI, GC, and RSI components.
# stats_summary() computes RSI internally if it has not been run yet.
print(derip_obj.stats_summary())
```


```python
# Identify the sequence with the greatest strand imbalance (largest |RSI|)
stats = derip_obj.summarize_stats()
most_imbalanced = stats.loc[stats['RSI'].abs().idxmax()]
print(
    f'Most strand-imbalanced sequence: {most_imbalanced["ID"]}\n'
    f'  RSI   = {most_imbalanced["RSI"]:+.3f}\n'
    f'  p_fwd = {most_imbalanced["p_fwd"]:.3f}   p_rev = {most_imbalanced["p_rev"]:.3f}'
)

# Rank the whole alignment from most forward- to most reverse-strand RIP.
# (Sequences with an undefined RSI sort to the end.)
print('\nSequences ranked by RSI (forward -> reverse):')
for rec in derip_obj.sort_by_rsi(descending=True):
    print(f'  {rec.id}: RSI = {rec.annotations["RSI"]:+.3f}')
```

## 12. Mutation spectra (SBS-96) from the deRIP consensus

deRIP2 can summarise every substitution in the alignment as a **single-base-substitution (SBS-96) mutation spectrum**, using the RIP-corrected consensus (`gapped_consensus`) as the ancestral reference. Each of the 96 channels is a pyrimidine-folded substitution (`C>A/G/T`, `T>A/C/G`) in one of 16 trinucleotide contexts. RIP shows up as a sharp `C>T` peak concentrated in `NCA` (CpA) contexts.

Here we contrast two ancestral references built from the **same** *Sahana* alignment:

- **RIP-like only** (`reaminate=False`): only RIP-context deamination is reverted in the ancestor, so non-RIP `C→T` changes remain and are counted as events.
- **All deamination** (`reaminate=True`): every `C→T` / `G→A` is reverted in the ancestor, so deamination is treated as ancestral and drops out of the spectrum.

For the full feature set (SBS-192, strand asymmetry, homoplasy, and the rigorous phylogenetic method) see the [Mutation spectra tutorial](tutorials/mutation-spectra.md).


```python
# Build a second deRIP on the SAME filtered Sahana alignment, but revert ALL
# deamination (RIP-context and not) when reconstructing the ancestor. The DeRIP
# constructor accepts a MultipleSeqAlignment directly, so we reuse sahana.alignment.
sahana_reaminate = DeRIP(sahana.alignment, reaminate=True)
sahana_reaminate.calculate_rip(label='sahana_reaminate')

print(f'RIP-like correction   corrected columns: {len(sahana.corrected_positions)}')
print(
    f'All-deamination        corrected columns: {len(sahana_reaminate.corrected_positions)}'
)
```


```python
# Plot both SBS-96 spectra and display them for comparison
from IPython.display import Image, display

# RIP-like-only ancestor (reaminate=False); ancestor = the RIP-corrected consensus
riponly_png = os.path.join(output_dir, 'sahana_sbs96_riponly.png')
sahana.plot_spectra(
    output_file=riponly_png,
    kind='96',
    title='Sahana SBS-96 (RIP-like correction)',
)

# All-deamination ancestor (reaminate=True)
reaminate_png = os.path.join(output_dir, 'sahana_sbs96_reaminate.png')
sahana_reaminate.plot_spectra(
    output_file=reaminate_png,
    kind='96',
    title='Sahana SBS-96 (all deamination corrected)',
)

print('RIP-like correction:')
display(Image(riponly_png))
print('All deamination corrected (--reaminate):')
display(Image(reaminate_png))

# The RIP-like spectrum is dominated by the C>T peak in NCA (CpA) contexts.
# Reaminating the ancestor removes deamination events from the reference, so the
# reaminated spectrum shifts toward the residual, non-deamination substitutions.
```

## Conclusion

This tutorial demonstrates how to use the `DeRIP` class directly for programmatic analysis of RIP mutations in DNA alignments. The class-based approach offers more flexibility and integration possibilities compared to the command-line interface.


Key functionalities covered:
- Loading and examining alignments
- RIP detection and correction
- CRI calculation and sequence filtering
- GC content analysis
- Alignment visualization
- Comparing different parameter settings
- Per-sequence strand-imbalance (RSI) statistics
- SBS-96 mutation spectra from the deRIP consensus

For more information and advanced usage, see the other tutorials:
- [RIP strand bias](tutorials/rip-strand-bias.md)
- [RIP masking & phylogeny](tutorials/rip-masking-phylogeny.md)
- [Mutation spectra](tutorials/mutation-spectra.md)

...or the [deRIP2 GitHub repository](https://github.com/adamtaranto/deRIP2).
