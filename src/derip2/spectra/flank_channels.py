"""
Channel bookkeeping for flanking-context spectra of RIP-like sites.

Where the SBS-96 model (see :mod:`derip2.spectra.channels`) classifies a
*substitution* by its 5'/3' trinucleotide context, this model classifies a
*dinucleotide site* by the single base one position upstream and one position
downstream — a 4 bp motif ``[up][center][down]`` with a **fixed** two-base centre.
Only the two flanks vary, giving ``4 x 4 = 16`` channels per site state.

Two site states are counted (each with a fixed centre after orientation folding):

- **Substrate** — the surviving RIP substrate dinucleotide, ``CpA`` read on the
  pyrimidine (forward) strand. Centre ``'CA'``. A reverse-strand substrate reads
  as ``TpG`` on the forward strand and is reverse-complemented back to ``CA``.
- **Product** — the RIP product dinucleotide ``TpA`` in a RIP-informative column.
  Centre ``'TA'`` (``TpA`` is its own reverse complement's centre).

**Orientation folding** reverse-complements a reverse-strand motif so every count
lands on the ``CA``/``TA``-equivalent channel. Reverse-complementing a 4 bp motif
``[up][X][Y][down]`` gives ``[comp(down)][comp(Y)][comp(X)][comp(up)]``: the two
flanks **swap sides and complement** (the same load-bearing detail as SBS-96's
pyrimidine fold). The centre ``CA`` <-> ``TG`` maps to ``CA`` and ``TA`` <-> ``TA``,
so both states keep a fixed centre after folding.

Channel order matches the SBS-96 flank convention: the upstream base varies in the
outer loop and the downstream base in the inner loop, so the 16 flanks are laid
out ``AA, AC, AG, AT, CA, ..., TT`` for a given centre (e.g. ``ACAA, ACAC, ACAG,
ACAT, CCAA, ..., TCAT`` for centre ``CA``).
"""

from typing import List, Tuple

import numpy as np

from derip2.spectra.channels import BASES, COMPLEMENT

# Base-code convention shared with :mod:`derip2.stats.mutation_spectra`:
# A=0, C=1, G=2, T=3 (the order of ``BASES``).
_CODE_TO_BASE: Tuple[str, ...] = BASES

# Complement lookup on base codes: A(0)<->T(3), C(1)<->G(2). Applying it to a code
# array is the vectorised equivalent of ``COMPLEMENT`` on characters, used by the
# reverse-strand fold in :mod:`derip2.stats.flank_spectra`.
COMP_CODE: np.ndarray = np.array(
    [BASES.index(COMPLEMENT[b]) for b in BASES], dtype=np.int64
)

# Channel index for a resolved ``(up_code, down_code)`` flank pair. With the
# upstream base outer and downstream base inner this is simply ``up*4 + down``,
# matching the SBS-96 flank ordering so plots and matrices line up 1:1.
IDX16_TABLE: np.ndarray = (
    np.arange(4, dtype=np.int64)[:, None] * 4 + np.arange(4, dtype=np.int64)[None, :]
)

# The two centre dinucleotides, keyed by site state.
CENTER_SUBSTRATE: str = 'CA'
CENTER_PRODUCT: str = 'TA'


def flank_channel_labels(center: str) -> List[str]:
    """
    Enumerate the 16 ``[up][center][down]`` motif labels for a site state.

    The upstream base varies in the outer loop and the downstream base in the
    inner loop, so the returned order matches :data:`IDX16_TABLE` (channel
    ``up*4 + down``) and the SBS-96 flank convention.

    Parameters
    ----------
    center : str
        The fixed two-base centre, ``'CA'`` (substrate) or ``'TA'`` (product).

    Returns
    -------
    list of str
        The 16 four-base motif labels in canonical channel order, e.g.
        ``['ACAA', 'ACAC', ..., 'TCAT']`` for ``center='CA'``.

    Raises
    ------
    ValueError
        If ``center`` is not a two-character ``ACGT`` string.
    """
    if len(center) != 2 or any(b not in BASES for b in center):
        raise ValueError(f'center must be a two-base ACGT motif, got {center!r}')
    return [f'{up}{center}{down}' for up in BASES for down in BASES]


def flank_pair_labels() -> List[str]:
    """
    Enumerate the 16 centre-agnostic ``up.down`` flank-pair labels.

    These label the flank context alone (no centre dinucleotide), for comparing a
    substrate (``CA``-centred) spectrum against a product (``TA``-centred) one
    position-by-position without implying a shared centre motif.

    Returns
    -------
    list of str
        The 16 flank-pair labels in channel order, e.g.
        ``['A.A', 'A.C', ..., 'T.T']``.
    """
    return [f'{up}.{down}' for up in BASES for down in BASES]


# Canonical label sets, precomputed once.
FLANK16_LABELS_CA: List[str] = flank_channel_labels(CENTER_SUBSTRATE)
FLANK16_LABELS_TA: List[str] = flank_channel_labels(CENTER_PRODUCT)
FLANK16_PAIR_LABELS: List[str] = flank_pair_labels()
