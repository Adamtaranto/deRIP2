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

# Only the light, dependency-free helpers are re-exported at the package level.
# The phylogenetic submodules (``tree_asr``, ``call_mutations``, ``qc``) pull in
# ete4/pandas and import back into ``derip2.stats.mutation_spectra``; importing
# them here would create a circular import, so callers import them directly, e.g.
# ``from derip2.spectra.tree_asr import reconstruct``.
from derip2.spectra.channels import (
    DOWNSTREAM_CHANNELS,
    SBS96_CHANNELS,
    SBS192_CHANNELS,
    downstream_channel,
    downstream_context,
    fold_to_pyrimidine,
    parse_downstream_channel,
    sbs96_channel,
    sbs192_channel,
    trinucleotide_context,
)
from derip2.spectra.matrix_io import (
    read_sbs_matrix,
    write_matrix_metadata,
    write_sbs_matrix,
)

__all__ = [
    'DOWNSTREAM_CHANNELS',
    'SBS96_CHANNELS',
    'SBS192_CHANNELS',
    'downstream_channel',
    'downstream_context',
    'fold_to_pyrimidine',
    'parse_downstream_channel',
    'sbs192_channel',
    'sbs96_channel',
    'trinucleotide_context',
    'read_sbs_matrix',
    'write_matrix_metadata',
    'write_sbs_matrix',
]
