[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# deRIP2

Predict progenitor sequence of fungal repeat families by correcting for RIP-like mutations 
(CpA --> TpA) and cytosine deamination (C --> T) events.

Mask RIP or deamination events from input alignment as ambiguous bases.

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
  - If proportion of positions in column in RIP-like context => minRIPlike threshold, AND at least one substrate and one product motif (i.e. CpA and TpA) is present, perform RIP correction in output sequence.
  - For all remaining positions in output sequence (not filled by gap, reaminate, or RIP-correction) inherit sequence from input sequence with the fewest observed RIP events (or greatest GC content if not RIP detected or multiple sequences sharing min-RIP count).

Outputs:
  - Corrected sequence as fasta.
  - Optional, alignment with: 
    - Corrected sequence appended.
    - Corrected positions masked as ambiguous bases.
  

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
derip2 0.0.3

# Get usage information
% derip2 --help
```

### Example usage

For aligned sequences in 'myalignment.fa':
  - Any column >= 70% gap positions is not corrected.
  - Bases in column must be >= 80% C/T or G/A 
  - At least 50% bases must be in RIP dinucleotide context (C/T as CpA / TpA)
  - Inherit all remaining uncorrected positions from least RIP'd sequence.
  - Mask all substrate and product motifs from corrected columns as ambiguous bases (i.e. CpA to TpA --> YpA)

```bash
derip2 --inAln myalignment.fa --format fasta \
--maxGaps 0.7 \
--maxSNPnoise 0.2 \
--minRIPlike 0.5 \
--outDir results \
--outAlnName aligment_with_deRIP.fa \
--label deRIPseqName \
--mask > results/deRIPed_sequence.fa
```

**Output:**  
  - results/deRIPed_sequence.fa
  - results/masked_aligment_with_deRIP.fa

### Standard options

```
Usage: derip2 [-h] [--version] -i INALN
              [--format {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}]
              [-g MAXGAPS] [-a] [--maxSNPnoise MAXSNPNOISE]
              [--minRIPlike MINRIPLIKE] [--fillmaxgc] [--fillindex FILLINDEX]
              [--mask] [--noappend] [-d OUTDIR] [--outAlnName OUTALNNAME]
              [--outAlnFormat {fasta,nexus}] [--label LABEL]

Predict ancestral sequence of fungal repeat elements by correcting for RIP-
like mutations or cytosine deamination in multi-sequence DNA alignments.
Optionally, mask corrected positions in alignment.

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -i INALN, --inAln INALN
                        Multiple sequence alignment.
  --format {clustal,emboss,fasta,fasta-m10,ig,nexus,phylip,phylip-sequential,phylip-relaxed,stockholm}
                        Format of input alignment. Default: fasta
  -g MAXGAPS, --maxGaps MAXGAPS
                        Maximum proportion of gapped positions in column to be
                        tolerated before forcing a gap in final deRIP
                        sequence. Default: 0.7
  -a, --reaminate       Correct all deamination events independent of RIP
                        context. Default: False
  --maxSNPnoise MAXSNPNOISE
                        Maximum proportion of conflicting SNPs permitted
                        before excluding column from RIP/deamination
                        assessment. i.e. By default a column with >= 0.5 'C/T'
                        bases will have 'TpA' positions logged as RIP events.
                        Default: 0.5
  --minRIPlike MINRIPLIKE
                        Minimum proportion of deamination events in RIP
                        context (5' CpA 3' --> 5' TpA 3') required for column
                        to deRIP'd in final sequence. Note: If 'reaminate'
                        option is set all deamination events will be
                        corrected. Default 0.1
  --fillmaxgc           By default uncorrected positions in the output
                        sequence are filled from the sequence with the lowest
                        RIP count. If this option is set remaining positions
                        are filled from the sequence with the highest G/C
                        content. Default: False
  --fillindex FILLINDEX
                        Force selection of alignment row to fill uncorrected
                        positions from by row index number (indexed from 0).
                        Note: Will override '--fillmaxgc' option.
  --mask                Mask corrected positions in alignment with degenerate
                        IUPAC codes.
  --noappend            If set, do not append deRIP'd sequence to output
                        alignment.
  -d OUTDIR, --outDir OUTDIR
                        Directory for deRIP'd sequence files to be written to.
  --outAlnName OUTALNNAME
                        Optional: If set write alignment including deRIP
                        corrected sequence to this file.
  --outAlnFormat {fasta,nexus}
                        Optional: Write alignment including deRIP sequence to
                        file of format X. Default: fasta
  --label LABEL         Use label as name for deRIP'd sequence in output
                        files.
```

## Issues
Submit feedback to the [Issue Tracker](https://github.com/Adamtaranto/deRIP2/issues)

## License
Software provided under MIT license.