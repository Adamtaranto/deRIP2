"""
DeRIP2: Detection and Removal of RIP mutations from fungal genome sequences.

DeRIP2 is a tool for identifying and correcting Repeat-Induced Point (RIP) mutations
in fungal genome sequences. RIP is a genome defense mechanism in fungi that introduces
C-to-T mutations in duplicated sequences during the sexual cycle. This package provides
tools to detect these mutations and restore sequences to their pre-RIP state.

Notes
-----
RIP (Repeat-Induced Point mutation) is a fungal genome defense mechanism that
specifically targets duplicated DNA sequences by converting cytosine to thymine
nucleotides. This process typically occurs in a CpA dinucleotide context,
resulting in characteristic TpA dinucleotides in mutated sequences.

Examples
--------
>>> from derip2 import derip
>>> # Process sequences to detect and correct RIP mutations
>>> result = derip.process_alignment('input.fasta', 'output.fasta')
"""
