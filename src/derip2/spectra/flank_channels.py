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

from itertools import product
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


def _flank_strings(width: int) -> List[str]:
    """
    Enumerate every ``width``-base flank string in canonical (base-4) order.

    The first character is the most significant base, matching the mixed-radix
    channel index in :func:`flank_channel_index`. For ``width == 1`` this is just
    ``['A', 'C', 'G', 'T']``; for ``width == 2`` it is ``['AA', 'AC', ..., 'TT']``.

    Parameters
    ----------
    width : int
        Number of flanking bases on one side (>= 1).

    Returns
    -------
    list of str
        The ``4 ** width`` flank strings in channel order.
    """
    if width < 1:
        raise ValueError(f'width must be >= 1, got {width}')
    return [''.join(combo) for combo in product(BASES, repeat=width)]


def n_flank_channels(width: int = 1) -> int:
    """
    Number of flank channels for a given flank width.

    Parameters
    ----------
    width : int, optional
        Number of flanking bases on each side (default 1).

    Returns
    -------
    int
        The channel count, ``4 ** (2 * width)`` (16 for a 1 bp flank).
    """
    return 4 ** (2 * width)


def flank_grid_dim(width: int = 1) -> int:
    """
    Side length of the (upstream x downstream) channel grid.

    Parameters
    ----------
    width : int, optional
        Number of flanking bases on each side (default 1).

    Returns
    -------
    int
        The grid side length, ``4 ** width`` (4 for a 1 bp flank).
    """
    return 4**width


def flank_channel_labels(center: str, width: int = 1) -> List[str]:
    """
    Enumerate the ``[up][center][down]`` motif labels for a site state.

    The upstream flank (``width`` bases, written 5'->3') varies in the outer loop
    and the downstream flank in the inner loop, so the returned order matches the
    mixed-radix channel index of :func:`flank_channel_index`. For ``width == 1``
    this reproduces the 16-channel ``up*4 + down`` ordering exactly (e.g.
    ``['ACAA', 'ACAC', ..., 'TCAT']`` for ``center='CA'``).

    Parameters
    ----------
    center : str
        The fixed two-base centre, ``'CA'`` (substrate) or ``'TA'`` (product).
    width : int, optional
        Number of flanking bases on each side (default 1).

    Returns
    -------
    list of str
        The ``4 ** (2 * width)`` motif labels in canonical channel order.

    Raises
    ------
    ValueError
        If ``center`` is not a two-character ``ACGT`` string, or ``width < 1``.
    """
    if len(center) != 2 or any(b not in BASES for b in center):
        raise ValueError(f'center must be a two-base ACGT motif, got {center!r}')
    flanks = _flank_strings(width)
    return [f'{up}{center}{down}' for up in flanks for down in flanks]


def flank_pair_labels(width: int = 1) -> List[str]:
    """
    Enumerate the centre-agnostic ``up.down`` flank-pair labels.

    These label the flank context alone (no centre dinucleotide), for comparing a
    substrate (``CA``-centred) spectrum against a product (``TA``-centred) one
    position-by-position without implying a shared centre motif.

    Parameters
    ----------
    width : int, optional
        Number of flanking bases on each side (default 1).

    Returns
    -------
    list of str
        The ``4 ** (2 * width)`` flank-pair labels in channel order, e.g.
        ``['A.A', 'A.C', ..., 'T.T']`` for ``width == 1``.
    """
    flanks = _flank_strings(width)
    return [f'{up}.{down}' for up in flanks for down in flanks]


def flank_channel_index(
    up_nf: np.ndarray, down_nf: np.ndarray, width: int
) -> np.ndarray:
    """
    Map resolved (upstream, downstream) flank codes to channel indices.

    Parameters
    ----------
    up_nf, down_nf : numpy.ndarray
        ``(n_sites, width)`` int arrays of folded flank base codes in
        **nearest-first** order (column 0 is the base adjacent to the centre,
        column ``width - 1`` is the outermost).
    width : int
        Flank width (columns of ``up_nf`` / ``down_nf``).

    Returns
    -------
    numpy.ndarray
        ``(n_sites,)`` channel indices in ``[0, 4 ** (2 * width))``, consistent
        with :func:`flank_channel_labels`. The label writes the upstream flank
        outermost-base-first, so the upstream codes are reversed to label order
        before encoding; the two flanks are then a single base-4 number with the
        outermost upstream base most significant and the outermost downstream base
        least significant. For ``width == 1`` this is ``up * 4 + down``.
    """
    if up_nf.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    # Label order = [U_w..U_1, D_1..D_w]: reverse the upstream (nearest-first ->
    # outermost-first), keep downstream nearest-first.
    digits = np.concatenate([up_nf[:, ::-1], down_nf], axis=1)
    weights = (4 ** np.arange(2 * width - 1, -1, -1)).astype(np.int64)
    return digits.astype(np.int64) @ weights


# Canonical label sets, precomputed once (width 1; back-compat public constants).
FLANK16_LABELS_CA: List[str] = flank_channel_labels(CENTER_SUBSTRATE)
FLANK16_LABELS_TA: List[str] = flank_channel_labels(CENTER_PRODUCT)
FLANK16_PAIR_LABELS: List[str] = flank_pair_labels()
