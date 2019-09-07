[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# deRIP2

Predict progenitor sequence of fungal repeat families by correcting for RIP-like mutations 
(CpA --> TpA) and cytosine deamination (C --> T) events.

# Table of contents
* [Algorithm overview](#algorithm-overview)
* [Options and usage](#options-and-usage)
    * [Installation](#installation)
    * [Example usage](#example-usage)
    * [Standard options](#standard-options)
* [Issues](#issues)
* [License](#license)

## Algorithm overview

For each column in input alignment:
  - Check if number of gapped rows is greater than max gap proportion. If true, then a gap is added to the output sequence.
  - Set invariant column values in output sequence.
  - If at least X proportion of bases are C/T or G/A (i.e. maxSNPnoise = 0.4, then at least 0.6 of positions in column must be C/T or G/A).
  - If reaminate option is set then revert T-->C or A-->G.
  - If reaminate is not set then check for number of positions in RIP dinucleotide context (C/TpA or TpG/A).
  - If proportion of positions in column in RIP-like context => minRIPlike threshold, perform RIP correction in output sequence.
  - For all remaining positions in output sequence (not filled by gap, reaminate, or RIP-correction) inherit sequence from input sequence with the fewest observed RIP events (or greatest GC content if not RIP detected).

Outputs:
  - Alignment with corrected sequence appended.
  - Corrected sequence as fasta.

## Options and Usage

### Installation

Requires Python => v3.6

Clone from this repository:

```bash
% git clone https://github.com/Adamtaranto/deRIP2.git && cd deRIP2 && pip install -e .
```

Install from PyPi.

```bash
% pip install derip2
```

Test installation.

```bash
# Print version number and exit.
% derip2 --version
derip2 0.0.1

# Get usage information
% derip2 --help
```

### Example usage

For aligned sequences in 'myalignment.fa':
  - Any column >= 50% non RIP/cytosine deamination mutations is not corrected.
  - Any column >= 70% gap positions is not corrected.
  - Make RIP corrections if column is >= 10% RIP context.
  - Correct Cytosine-deamination mutations outside of RIP context.
  - Inherit all remaining uncorrected positions from least RIP'd sequence.

```bash
derip2 --inAln myalignment.fa --format fasta \
--maxGaps 0.7 \
--maxSNPnoise 0.5 \
--minRIPlike 0.1 \
--outName deRIPed_sequence.fa \
--outAlnName aligment_with_deRIP.aln \
--outAlnFormat clustal \
--label deRIPseqName \
--outDir results \
--reaminate
```

**Output:**  
  - results/aligment_with_deRIP.aln 

### Standard options

```
Usage: derip2 [-h] [--version] -i INALN
              [--format {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}]
              [--outAlnFormat {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}]
              [-g MAXGAPS] [-a] [--maxSNPnoise MAXSNPNOISE]
              [--minRIPlike MINRIPLIKE] [-o OUTNAME] [--outAlnName OUTALNNAME]
              [--label LABEL] [-d OUTDIR]

Predict ancestral sequence of fungal repeat elements by correcting for RIP-
like mutations in multi-sequence DNA alignments.

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -i INALN, --inAln INALN
                        Multiple sequence alignment.
  --format {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}
                        Format of input alignment.
  --outAlnFormat {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}
                        Optional: Write alignment including deRIP sequence to
                        file of format X.
  -g MAXGAPS, --maxGaps MAXGAPS
                        Maximum proportion of gapped positions in column to be
                        tolerated before forcing a gap in final deRIP sequence.
  -a, --reaminate       Correct deamination events in non-RIP contexts.
  --maxSNPnoise MAXSNPNOISE
                        Maximum proportion of conflicting SNPs permitted
                        before excluding column from RIP/deamination
                        assessment. i.e. By default a column with >= 0.5 'C/T'
                        bases will have 'TpA' positions logged as RIP events.
  --minRIPlike MINRIPLIKE
                        Minimum proportion of deamination events in RIP
                        context (5' CpA 3' --> 5' TpA 3') required for column
                        to be deRIP'd in final sequence. Note: If 'reaminate'
                        option is set all deamination events will be
                        corrected
  -o OUTNAME, --outName OUTNAME
                        Write deRIP'd sequence to this file.
  --outAlnName OUTALNNAME
                        Optional: If set, write alignment including deRIP
                        sequence to this file.
  --label LABEL         Use label as name for deRIP'd sequence in output
                        files.
  -d OUTDIR, --outDir OUTDIR
                        Directory for deRIP'd sequence files to be written to.
```

## Issues
Submit feedback to the [Issue Tracker](https://github.com/Adamtaranto/deRIP2/issues)

## License
Software provided under MIT license.