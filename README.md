<a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" align="left" height="20"/>
</a> 

<a href="https://gitpod.io/#https://github.com/adamtaranto/deRIP2">
  <img src="https://gitpod.io/button/open-in-gitpod.svg" align="right" height="35"/>
</a> 

<br clear="right"/>
<br clear="left"/>

# deRIP2

Predict progenitor sequence of fungal repeat families by correcting for RIP-like mutations 
(CpA --> TpA) and cytosine deamination (C --> T) events.

Mask RIP or deamination events from input alignment as ambiguous bases.

## Table of contents
- [Algorithm overview](#algorithm-overview)
- [Options and Usage](#options-and-usage)
  - [Installation](#installation)
  - [Example usage](#example-usage)
  - [Standard options](#standard-options)
- [Report Issues](#issues)
- [License](#license)

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
    - With corrected positions masked as ambiguous bases.

  

## Options and Usage

### Installation

Requires Python => v3.8

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

derip2 0.0.4


# Get usage information
% derip2 --help
```

### Example usage

For aligned sequences in 'mintest.fa':
  - Any column with >= 70% gap positions will not be corrected and a gap inserted in corrected sequence.
  - Bases in column must be >= 80% C/T or G/A 
  - At least 50% bases in a column must be in RIP dinucleotide context (C/T as CpA / TpA) for correction.
  - Default: Inherit all remaining uncorrected positions from the least RIP'd sequence.
  - Mask all substrate and product motifs from corrected columns as ambiguous bases (i.e. CpA to TpA --> YpA)

```bash
derip2 -i tests/data/mintest.fa --format fasta \
--maxGaps 0.7 \
--maxSNPnoise 0.2 \
--minRIPlike 0.5 \
--label derip_name \
--mask \
-d results \
--outAln masked_aligment_with_deRIP.fa --outAlnFormat fasta --outFasta derip_prediction.fa
```

**Output:**  
  - results/derip_prediction.fa
  - results/masked_aligment_with_deRIP.fa


## Issues
Submit feedback to the [Issue Tracker](https://github.com/Adamtaranto/deRIP2/issues)

## License
Software provided under MIT license.