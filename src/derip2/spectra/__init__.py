"""
Trinucleotide-context mutation spectrum analysis (SigProfiler SBS-96 / SBS-192).

This subpackage turns a deRIP2 alignment into single-base-substitution (SBS)
spectra: the count of every substitution in the context of its immediately
flanking 5' and 3' bases. It provides the channel bookkeeping (:mod:`channels`)
and SigProfiler-compliant matrix I/O (:mod:`matrix_io`) shared by the tree-free
baseline and, in a later milestone, the phylogenetic ancestral-reconstruction
path.

The spectrum computation itself lives in :mod:`derip2.stats.mutation_spectra`,
alongside the other per-alignment statistics.
"""

from derip2.spectra.channels import (
    SBS96_CHANNELS,
    SBS192_CHANNELS,
    fold_to_pyrimidine,
    sbs96_channel,
    sbs192_channel,
    trinucleotide_context,
)
from derip2.spectra.matrix_io import read_sbs_matrix, write_sbs_matrix

__all__ = [
    'SBS96_CHANNELS',
    'SBS192_CHANNELS',
    'fold_to_pyrimidine',
    'sbs192_channel',
    'sbs96_channel',
    'trinucleotide_context',
    'read_sbs_matrix',
    'write_sbs_matrix',
]
