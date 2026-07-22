[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PyPI version](https://badge.fury.io/py/derip2.svg)](https://badge.fury.io/py/derip2)
[![codecov](https://codecov.io/gh/adamtaranto/derip2/branch/main/graph/badge.svg)](https://codecov.io/gh/adamtaranto/derip2)
[![BioConda Install](https://img.shields.io/conda/dn/bioconda/derip2.svg?style=flag&label=BioConda%20install)](https://anaconda.org/bioconda/derip2)

```code
██████╗ ███████╗██████╗ ██╗██████╗ ██████╗
██╔══██╗██╔════╝██╔══██╗██║██╔══██╗╚════██╗
██║  ██║█████╗  ██████╔╝██║██████╔╝ █████╔╝
██║  ██║██╔══╝  ██╔══██╗██║██╔═══╝ ██╔═══╝
██████╔╝███████╗██║  ██║██║██║     ███████╗
╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚══════╝
```

deRIP2 scans aligned sequences for evidence of un-RIP'd precursor states, allowing
for improved RIP-correction across large repeat families in which members are
independently RIP'd.

Use deRIP2 to:

- Predict ancestral fungal transposon sequences by correcting for RIP-like mutations
  (CpA --> TpA) and cytosine deamination (C --> T) events.

- Mask RIP or deamination events as ambiguous bases to remove RIP signal from phylogenetic analyses.

- Analyse mutation spectra in aligned sequences for evidence of RIP or other mutational processes.

- Generate an interactive per-sequence HTML report, and — with a GFF3 gene model
  — report how RIP alters the encoded protein (premature stops, non-synonymous
  changes, frameshifts, broken splice sites).

## Table of contents

- [Installation](#installation)
- [Example usage](#example-usage)
- [Per-sequence reporting](#per-sequence-reporting)
- [Standard Options](#standard-options)
- [Spectra Options](#spectra-options)
- [Algorithm overview](#algorithm-overview)
- [License](#license)

## Installation

Install from PyPi.

```bash
pip install derip2
```

Install from Bioconda.

```bash
#iqtree is an optional dependency for 'derip2-spectra'
conda install -c bioconda derip2 iqtree
```

Pip install latest development version from GitHub.

```bash
pip install git+https://github.com/Adamtaranto/deRIP2.git
```

## Example usage

See the [DeRIP2 tutorials pages](https://adamtaranto.github.io/deRIP2) for full usage documentation.

### Basic usage with masking

The `--mask` option outputs an alignment with all RIP-like C/T positions masked as 'Y'. Use this to generate phylogenies free of RIP influence.

For aligned sequences in 'mintest.fa':

- Any column with >= 70% gap positions will not be corrected and a gap inserted in corrected sequence.
- Bases in column must be >= 80% C/T or G/A
- At least 50% bases in a column must be in RIP dinucleotide context (C/T as CpA / TpA) for correction.
- Default: Inherit all remaining uncorrected positions from the least RIP'd sequence.
- Mask all substrate and product motifs from corrected columns as ambiguous bases (i.e. CpA to TpA --> YpA)

```bash
derip2 -i tests/data/mintest.fa \
  --max-gaps 0.7 \
  --max-snp-noise 0.2 \
  --min-rip-like 0.5 \
  --mask \
  -d results \
  --prefix derip_output
```

**Output:**

- `results/derip_output.fasta` - Corrected sequence
- `results/derip_output_masked_alignment.fasta` - Alignment with masked corrections

### With vizualization

The `--plot` option will create a visualization of the alignment with RIP markup. The `--plot-rip-type` option can be used to specify the type of RIP events to be displayed in the alignment visualization `product`, `substrate`, or `both`.

```bash
derip2 -i tests/data/mintest.fa \
  --max-gaps 0.7 \
  --max-snp-noise 0.2 \
  --min-rip-like 0.5 \
  --plot \
  --plot-rip-type both \
  -d results \
  --prefix derip_output
```

**Output:**

- `results/derip_output.fasta` - Corrected sequence
- `results/derip_output_masked_alignment.fasta` - Alignment with masked corrections
- `results/derip_output_visualization.svg` - Visualization of the alignment with RIP markup

![Visualization of the alignment with RIP markup](https://raw.githubusercontent.com/Adamtaranto/deRIP2/main/docs/img/derip_output_visualization.svg)

### Using maximum GC content for filling

By default uncorrected positions in the output sequence are filled from the sequence with the lowest RIP count. If the `--fill-max-gc` option is set, remaining positions are filled from the sequence with the highest G/C content sequence instead.

```bash
derip2 -i tests/data/mintest.fa \
  --max-gaps 0.7 \
  --max-snp-noise 0.2 \
  --min-rip-like 0.5 \
  --fill-max-gc \
  -d results \
  --prefix derip_gc_filled
```

Alternatively, the `--fill-index` option can be used to force selection of alignment row to fill uncorrected positions from by row index number (indexed from 0). Note: This will override the `--fill-max-gc` option.

### Correcting all deamination events

If the `--reaminate` option is set, all deamination events will be corrected, regardless of RIP context.

`--plot-rip-type product` is used to highlight the product of RIP events in the visualization.
Non-RIP deamination events are also highlighted.

```bash
derip2 -i tests/data/mintest.fa \
  --max-gaps 0.7 \
  --reaminate \
  -d results \
  --plot \
  --plot-rip-type product \
  --prefix derip_reaminated
```

**Output:**

- `results/derip_reaminated.fasta` - Corrected sequence using highest GC content sequence for filling
- `results/derip_reaminated_alignment.fasta` - Alignment with corrected sequence appended
- `results/derip_reaminated_vizualization.svg` - Visualization of the alignment with RIP markup

![Visualization of the alignment with RIP markup](https://raw.githubusercontent.com/Adamtaranto/deRIP2/main/docs/img/derip_reaminated_visualization.svg)

### Mutation spectra (`derip2-spectra`)

`derip2-spectra` builds trinucleotide-context **SBS-96** and **SBS-192** mutation
spectra (SigProfiler-compliant matrices plus plots), so RIP (a `C>T` peak in CpA
context) can be told apart from other cytosine-deamination processes. It offers a
fast tree-free baseline and a phylogenetic path (IQ-TREE ancestral reconstruction)
that counts recurrent deamination correctly.

```bash
# Tree-free baseline (no external tools)
derip2-spectra -i tests/data/mintest.fa -d results -p family

# CHG-aware downstream-triplet context: classify each substitution by the mutated
# base plus its two downstream bases (motif ref-d1-d2), so methylation-driven C>T
# in the fungal CHG context becomes visible (writes family.SBSdownstream.txt).
derip2-spectra -i family.fasta --context downstream -d results -p family
```

![SBS-96 mutation spectrum of a RIP-affected transposon](https://raw.githubusercontent.com/Adamtaranto/deRIP2/main/docs/img/spectra_sbs96.png)

![Downstream-triplet spectrum of a RIP-affected transposon](https://raw.githubusercontent.com/Adamtaranto/deRIP2/main/docs/img/spectra_downstream.png)

See the [Mutation Spectra tutorial](https://adamtaranto.github.io/deRIP2/tutorials/mutation-spectra/)
for the full walkthrough, including supplying your own phylogeny and per-group
spectra.

## Per-sequence reporting

The `--per-seq-report` option writes a single self-contained
`prefix_per_sequence.html` with one arrow-key-navigable panel per input
sequence: the alignment row with RIP sites highlighted, a fixed-height
per-sequence strand-bias strip, a per-sequence SBS-96 spectrum against the
reconstructed ancestor, and that sequence's statistics.

```bash
derip2 -i tests/data/mintest.fa --per-seq-report -d results
```

Supply a GFF3 gene model with `--gff` (sequence ids must match the alignment;
coordinates are auto-adjusted for gaps) to add a gene-annotation track to
`--plot`, gene-effect panels (premature stops, non-synonymous changes,
frameshifts, broken splice sites) plus the deRIP-restored protein to the report,
and a `prefix_snp_effects.txt` summary.

```bash
derip2 -i tests/data/mintest.fa \
  --gff tests/data/mintest.gff3 \
  --per-seq-report --plot -d results
```

![Alignment with gene-annotation track](https://raw.githubusercontent.com/Adamtaranto/deRIP2/main/docs/img/annotation_track.svg)

See the [Per-sequence Reporting tutorial](https://adamtaranto.github.io/deRIP2/tutorials/per-sequence-reporting/)
for the full walkthrough.

## Standard options

```code
Usage: derip2 [OPTIONS]

  Predict ancestral sequence of fungal repeat elements by correcting for RIP-
  like mutations or cytosine deamination in multi-sequence DNA alignments.
  Optionally, mask mutated positions in alignment.

Options:
  --version                       Show the version and exit.
  -i, --input TEXT                Multiple sequence alignment.  [required]
  -g, --max-gaps FLOAT            Maximum proportion of gapped positions in
                                  column to be tolerated before forcing a gap
                                  in final deRIP sequence.  [default: 0.7]
  -a, --reaminate                 Correct all deamination events independent
                                  of RIP context.
  --max-snp-noise FLOAT           Maximum proportion of conflicting SNPs
                                  permitted before excluding column from
                                  RIP/deamination assessment. i.e. By default
                                  a column with >= 0.5 'C/T' bases will have
                                  'TpA' positions logged as RIP events.
                                  [default: 0.5]
  --min-rip-like FLOAT            Minimum proportion of deamination events in
                                  RIP context (5' CpA 3' --> 5' TpA 3')
                                  required for column to deRIP'd in final
                                  sequence. Note: If 'reaminate' option is set
                                  all deamination events will be corrected.
                                  [default: 0.1]
  --fill-max-gc                   By default uncorrected positions in the
                                  output sequence are filled from the sequence
                                  with the lowest RIP count. If this option is
                                  set remaining positions are filled from the
                                  sequence with the highest G/C content.
  --fill-index INTEGER            Force selection of alignment row to fill
                                  uncorrected positions from by row index
                                  number (indexed from 0). Note: Will override
                                  '--fill-max-gc' option.
  --mask                          Mask corrected positions in alignment with
                                  degenerate IUPAC codes.
  --no-append                     If set, do not append deRIP'd sequence to
                                  output alignment.
  -d, --out-dir TEXT              Directory for deRIP'd sequence files to be
                                  written to.
  -p, --prefix TEXT               Prefix for output files. Output files will
                                  be named prefix.fasta,
                                  prefix_alignment.fasta, etc.  [default:
                                  deRIPseq]
  --plot                          Create a visualization of the alignment with
                                  RIP markup.
  --plot-rip-type [both|product|substrate]
                                  Specify the type of RIP events to be
                                  displayed in the alignment visualization.
                                  [default: both]
  --plot-strand-bias              Create a diverging stacked-bar chart of per-
                                  column RIP strand bias.
  --strand-bias-scale [column|alignment|counts]
                                  Bar height normalisation: each column to its
                                  own depth, to the number of sequences (so
                                  gappy columns are short), or raw counts.
                                  [default: column]
  --strand-bias-xaxis [none|logo|derip]
                                  Draw a sequence logo or the deRIP'd
                                  consensus along the zero line.  [default:
                                  none]
  --strand-bias-columns [rip|substrate|all]
                                  Which positions are lettered along the zero
                                  line: RIP-like columns and their
                                  dinucleotide partners, untouched substrate
                                  columns and their partners, or every
                                  position. Every column is drawn as a bar
                                  regardless. Only has an effect with
                                  --strand-bias-xaxis logo or derip.
                                  [default: all]
  --strand-bias-stack [signal|product|all]
                                  Which bases each bar is made of: the RIP
                                  product and its unmutated substrate, the
                                  product alone, or every base with the
                                  remainder drawn translucent. Bars are never
                                  rescaled, so the missing height shows what
                                  was excluded.  [default: signal]
  --rsi-ambiguous [split|exclude|weight|both]
                                  How to attribute a TA dinucleotide that
                                  could have arisen from RIP on either strand
                                  when calculating RSI.  [default: split]
  --sort-by-rsi                   Sort the output alignment from most forward-
                                  to most reverse-strand RIP.
  --stats-out                     Write the per-sequence statistics table to
                                  prefix_stats.tsv.
  --html-report                   Write a self-contained HTML report to
                                  prefix_report.html.
  --per-seq-report                Write an interactive per-sequence HTML
                                  report to prefix_per_sequence.html (one
                                  arrow-key-navigable panel per sequence).
  --max-report-seqs INTEGER       Cap the number of sequence panels in the
                                  per-sequence report. When the alignment has
                                  more sequences, the strongest strand-bias
                                  sequences are kept. Unset renders every
                                  sequence.
  --gff TEXT                      GFF3 gene model. Sequence ids must match
                                  alignment record ids. Enables a gene-
                                  annotation track on --plot, gene-effect
                                  panels in the per-sequence report, and a
                                  prefix_snp_effects.txt summary.
  --genetic-code INTEGER          NCBI genetic code table for CDS translation
                                  and effect prediction.  [default: 1]
  --annotation-colors TEXT        Two-column (type<TAB>hex) file overriding
                                  default annotation-track colours by feature
                                  type.
  --loglevel [DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                  Set logging level.  [default: INFO]
  --logfile TEXT                  Log file path.
  -h, --help                      Show this message and exit.
  ```

## Spectra Options

```code
 Usage: derip2-spectra [OPTIONS]

  Build SBS-96 and SBS-192 trinucleotide mutation spectra from a DNA alignment
  by calling substitutions against the deRIP'd ancestral consensus, or via IQ-
  TREE ancestral reconstruction (--method phylo).

Options:
  --version                       Show the version and exit.
  -i, --input TEXT                Multiple sequence alignment (FASTA,
                                  optionally gzipped).  [required]
  -d, --out-dir TEXT              Directory for spectrum output files.
  -p, --prefix TEXT               Prefix for output files.  [default:
                                  deRIPspectra]
  --ancestor TEXT                 Optional FASTA of a hypothetical ancestor to
                                  call against instead of the reconstructed
                                  deRIP consensus. Must be the same length as
                                  the alignment.
  --reference-tag TEXT            Exact sequence ID of a pre-computed
                                  ancestral reference already present in the
                                  input alignment (e.g. a deRIP consensus you
                                  appended with derip2). When found (baseline
                                  method), that row is used as the ancestor
                                  and excluded from the counted sequences
                                  instead of re-running deRIP. Overridden by
                                  --ancestor.  [default: deRIPseq]
  --context [trinucleotide|downstream]
                                  Sequence context to classify substitutions
                                  by: the 5'/3' trinucleotide flanks
                                  (SBS-96/192), or the mutated base plus its
                                  two downstream bases (pyrimidine-folded
                                  96-channel, CHG-aware). The downstream
                                  context produces a single folded matrix, so
                                  --sbs 192/both do not apply.  [default:
                                  trinucleotide]
  --sbs [96|192|both]             Which SBS matrices/plots to produce
                                  (trinucleotide context only).  [default:
                                  both]
  --partition-by [none|row|clade]
                                  Split spectra into one pooled sample, one
                                  per sequence (baseline) or one per root
                                  clade (phylo).  [default: none]
  --groups TEXT                   Path to a two-column (name, group) file
                                  mapping sequences to group labels (e.g.
                                  species). Reports one spectrum per group;
                                  works for both methods and tolerates IQ-TREE
                                  name reformatting. Overrides --partition-by.
  --percentage                    Plot spectra as a percentage of each sample
                                  total.
  --min-hits INTEGER              Minimum independent hits for a site in the
                                  homoplasy report.  [default: 2]
  --no-plots                      Write matrices and tables only; skip
                                  figures.
  --method [baseline|phylo]       Spectrum method: tree-free single-reference
                                  baseline, or phylogenetic branch-by-branch
                                  calling via IQ-TREE ancestral
                                  reconstruction.  [default: baseline]
  --tree TEXT                     Fixed Newick tree for the phylo path; IQ-
                                  TREE reconstructs ancestral states on this
                                  topology instead of inferring a new tree.
  --iqtree-model TEXT             Substitution model passed to IQ-TREE (-m)
                                  for the phylo path.  [default: MFP]
  --threads TEXT                  IQ-TREE thread count (-T). AUTO benchmarks
                                  the best value; pass an integer to skip the
                                  benchmark (faster on small alignments).
                                  [default: AUTO]
  --rooting [midpoint|outgroup|none]
                                  How to root the tree for the phylo path
                                  (sets substitution direction).  [default:
                                  midpoint]
  --outgroup TEXT                 Outgroup tip name(s) for --rooting outgroup;
                                  comma-separate a clade.
  --min-prob FLOAT                Drop phylo events whose parent x child
                                  ancestral posterior is below this threshold.
                                  [default: 0.0]
  --root-sensitivity              Also report the fraction of edges whose
                                  direction flips under midpoint rooting
                                  (phylo path).
  -g, --max-gaps FLOAT            Maximum gap proportion in a column before it
                                  is gapped in the consensus.  [default: 0.7]
  -a, --reaminate                 Correct all deamination events regardless of
                                  RIP context when building the ancestor.
  --max-snp-noise FLOAT           Maximum proportion of conflicting SNPs
                                  before a column is excluded from RIP
                                  assessment.  [default: 0.5]
  --min-rip-like FLOAT            Minimum proportion of RIP-context
                                  deamination for a column to be corrected.
                                  [default: 0.1]
  --fill-max-gc                   Fill uncorrected positions from the highest-
                                  GC sequence rather than the least-RIP'd one.
  --fill-index INTEGER            Force the fill row by index (overrides
                                  --fill-max-gc).
  --loglevel [DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                  Set logging level.  [default: INFO]
  --logfile TEXT                  Log file path.
  -h, --help                      Show this message and exit.
```

## Algorithm overview

For each column in input alignment:

- Check if number of gapped rows is greater than max gap proportion. If true, then a gap is added to the output sequence.
- Set invariant column values in output sequence.
- If at least X proportion of bases are C/T or G/A (i.e. `max-snp-noise` = 0.4, then at least 0.6 of positions in column must be C/T or G/A).
- If reaminate option is set then revert T-->C or A-->G.
- If reaminate is not set then check for number of positions in RIP dinucleotide context (C/TpA or TpG/A).
- If proportion of positions in column in RIP-like context => `min-rip-like` threshold, AND at least one substrate and one product motif (i.e. CpA and TpA) is present, perform RIP correction in output sequence.
- For all remaining positions in output sequence (not filled by gap, reaminate, or RIP-correction) inherit sequence from input sequence with the fewest observed RIP events (or greatest GC content if RIP is not detected or multiple sequences sharing min-RIP count).

## License

Software provided under GPL-3 license.
