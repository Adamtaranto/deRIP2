[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/derip2.svg)](https://badge.fury.io/py/derip2)
[![codecov](https://codecov.io/gh/adamtaranto/derip2/branch/main/graph/badge.svg)](https://codecov.io/gh/adamtaranto/derip2)

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

## Table of contents

- [Installation](#installation)
- [Algorithm overview](#algorithm-overview)
- [Report Issues](#issues)
- [License](#license)

## Installation

Install from PyPi.

```bash
pip install derip2
```

Pip install latest development version from GitHub.

```bash
pip install git+https://github.com/Adamtaranto/deRIP2.git
```

Test installation.

```bash
# Print version number and exit.
derip2 --version

# Get usage information
derip2 --help
```

### Setup Development Environment

If you want to contribute to the project or run the latest development version, you can clone the repository and install the package in editable mode.

```bash
# Clone repository
git clone https://github.com/Adamtaranto/deRIP2.git && cd deRIP2

# Create virtual environment
conda env create -f environment.yml

# Activate environment
conda activate derip2-dev

# Install package in editable mode
pip install -e '.[dev]'
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

## Issues

Submit feedback and questions in the [discussion forum](https://github.com/Adamtaranto/deRIP2/discussions)

## License

Software provided under MIT license.
