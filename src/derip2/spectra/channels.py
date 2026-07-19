"""
SBS-96 and SBS-192 channel bookkeeping for trinucleotide mutation spectra.

The single-base-substitution (SBS) classification counts each substitution in the
context of its immediately flanking 5' and 3' bases.

**SBS-96** collapses every event onto the *pyrimidine* strand: the six pyrimidine
substitution types (``C>A, C>G, C>T, T>A, T>C, T>G``) times the sixteen 5'/3'
flank combinations. A purine-reference event is folded to its pyrimidine
complement by reverse-complementing the whole trinucleotide, which swaps *and*
complements the two flanks (the load-bearing correctness detail).

**SBS-192** is strand-resolved: it keeps the reference base as observed on a
defined reference strand (here the coding sense strand), so all twelve
substitution types times sixteen flanks = 192 channels are retained. This
exposes strand asymmetries (e.g. APOBEC/AID or transcription-coupled biases) that
the collapsed SBS-96 hides. This is the plain purine+pyrimidine channel form
described in the design (labels such as ``A[G>T]A``), not the transcriptional
``T:``/``U:`` prefixed form some SigProfiler versions emit for 192.

Channel label form is the SigProfiler convention ``5[REF>ALT]3`` (e.g.
``A[C>A]A``). ``SBS96_CHANNELS`` and ``SBS192_CHANNELS`` give the canonical row
order used by the matrix files and plots.
"""

import re
from typing import List, Optional, Tuple

import numpy as np

# Canonical base alphabet, ordered as SigProfiler orders flanks.
BASES: Tuple[str, ...] = ('A', 'C', 'G', 'T')
PYRIMIDINES: Tuple[str, ...] = ('C', 'T')
PURINES: Tuple[str, ...] = ('A', 'G')

# Watson-Crick complement, uppercase DNA only.
COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

# The six pyrimidine-reference substitution types, in SigProfiler order.
SBS96_SUBSTITUTIONS: Tuple[Tuple[str, str], ...] = (
    ('C', 'A'),
    ('C', 'G'),
    ('C', 'T'),
    ('T', 'A'),
    ('T', 'C'),
    ('T', 'G'),
)

# All twelve substitution types for the strand-resolved SBS-192 model: the six
# pyrimidine types above followed by their reverse-complement purine partners in
# the matching order (C>A -> G>T, C>G -> G>C, C>T -> G>A, T>A -> A>T,
# T>C -> A>G, T>G -> A>C).
SBS192_SUBSTITUTIONS: Tuple[Tuple[str, str], ...] = SBS96_SUBSTITUTIONS + tuple(
    (COMPLEMENT[ref], COMPLEMENT[alt]) for ref, alt in SBS96_SUBSTITUTIONS
)


def revcomp_base(base: str) -> str:
    """
    Return the Watson-Crick complement of a single uppercase DNA base.

    Parameters
    ----------
    base : str
        A single character, one of ``A``, ``C``, ``G`` or ``T``.

    Returns
    -------
    str
        The complementary base.

    Raises
    ------
    KeyError
        If ``base`` is not one of the four canonical DNA bases.
    """
    return COMPLEMENT[base]


def _channel_label(five: str, ref: str, alt: str, three: str) -> str:
    """
    Build a ``5[REF>ALT]3`` channel label.

    Parameters
    ----------
    five : str
        The 5' flanking base.
    ref : str
        The reference (ancestral) base.
    alt : str
        The derived base.
    three : str
        The 3' flanking base.

    Returns
    -------
    str
        The channel label, e.g. ``A[C>A]A``.
    """
    return f'{five}[{ref}>{alt}]{three}'


def _build_channels(
    substitutions: Tuple[Tuple[str, str], ...],
) -> List[str]:
    """
    Enumerate channel labels for a set of substitution types.

    The 5' base varies in the outer loop and the 3' base in the inner loop, so
    the sixteen flanks of each substitution type are contiguous. This matches the
    canonical SigProfiler row order.

    Parameters
    ----------
    substitutions : tuple of tuple of str
        Ordered ``(ref, alt)`` substitution types.

    Returns
    -------
    list of str
        Channel labels in canonical order.
    """
    channels: List[str] = []
    for ref, alt in substitutions:
        for five in BASES:
            for three in BASES:
                channels.append(_channel_label(five, ref, alt, three))
    return channels


# Canonical channel orderings and their reverse lookups.
SBS96_CHANNELS: List[str] = _build_channels(SBS96_SUBSTITUTIONS)
SBS192_CHANNELS: List[str] = _build_channels(SBS192_SUBSTITUTIONS)
SBS96_INDEX = {label: i for i, label in enumerate(SBS96_CHANNELS)}
SBS192_INDEX = {label: i for i, label in enumerate(SBS192_CHANNELS)}


def fold_to_pyrimidine(
    five: str, ref: str, alt: str, three: str
) -> Tuple[str, str, str, str]:
    """
    Fold a substitution onto the pyrimidine strand for SBS-96.

    If the reference base is already a pyrimidine (``C`` or ``T``) the event is
    returned unchanged. If it is a purine, the whole trinucleotide is
    reverse-complemented: the reference and derived bases are complemented, and
    the two flanks are complemented **and swapped** (the old 3' becomes the new
    5'). For example ``G>A`` in context ``T_C`` (5'=T, 3'=C) folds to ``C>T`` in
    context ``G_A``.

    Parameters
    ----------
    five : str
        The 5' flanking base on the reference strand.
    ref : str
        The reference (ancestral) base.
    alt : str
        The derived base.
    three : str
        The 3' flanking base on the reference strand.

    Returns
    -------
    tuple of str
        ``(five, ref, alt, three)`` on the pyrimidine strand.
    """
    if ref in PYRIMIDINES:
        return five, ref, alt, three
    # Purine reference: reverse-complement the trinucleotide. The flanks swap
    # sides because reversing the strand reverses 5'/3' orientation.
    return (
        COMPLEMENT[three],
        COMPLEMENT[ref],
        COMPLEMENT[alt],
        COMPLEMENT[five],
    )


def sbs96_channel(five: str, ref: str, alt: str, three: str) -> str:
    """
    Return the SBS-96 (pyrimidine-collapsed) channel label for an event.

    Parameters
    ----------
    five : str
        The 5' flanking base on the reference strand.
    ref : str
        The reference (ancestral) base.
    alt : str
        The derived base.
    three : str
        The 3' flanking base on the reference strand.

    Returns
    -------
    str
        The folded channel label, e.g. ``A[C>A]A``.
    """
    f5, r, a, f3 = fold_to_pyrimidine(five, ref, alt, three)
    return _channel_label(f5, r, a, f3)


def sbs192_channel(five: str, ref: str, alt: str, three: str) -> str:
    """
    Return the strand-resolved SBS-192 channel label for an event.

    No pyrimidine folding is applied: the reference base is kept as observed on
    the coding reference strand, so all twelve substitution types are retained.

    Parameters
    ----------
    five : str
        The 5' flanking base on the reference strand.
    ref : str
        The reference (ancestral) base.
    alt : str
        The derived base.
    three : str
        The 3' flanking base on the reference strand.

    Returns
    -------
    str
        The unfolded channel label, e.g. ``A[G>T]A``.
    """
    return _channel_label(five, ref, alt, three)


def trinucleotide_context(
    seq: np.ndarray,
    col: int,
    next_idx: np.ndarray,
    prev_idx: np.ndarray,
) -> Optional[Tuple[str, str]]:
    """
    Resolve the 5' and 3' flanking bases of a column using nearest non-gap bases.

    The flanks are read from ``seq`` at the nearest non-gap (and non-ambiguous)
    neighbour columns, whose indices are supplied precomputed in ``next_idx`` and
    ``prev_idx``. When either neighbour is absent (a terminal column, indicated
    by ``-1``) the event cannot be assigned a full trinucleotide context and
    ``None`` is returned.

    Parameters
    ----------
    seq : numpy.ndarray
        1-D byte array (dtype ``'S1'``) of the reference/ancestral sequence, with
        every non-``ACGT`` position already normalised to a gap so that the
        neighbour indices skip over it.
    col : int
        The column whose flanks are wanted.
    next_idx : numpy.ndarray
        1-D int array; the column index of the nearest non-gap base to the right
        of each column, or ``-1``.
    prev_idx : numpy.ndarray
        1-D int array; the column index of the nearest non-gap base to the left
        of each column, or ``-1``.

    Returns
    -------
    tuple of str or None
        ``(five, three)`` uppercase bases, or ``None`` if either flank is
        unresolvable.
    """
    left = int(prev_idx[col])
    right = int(next_idx[col])
    if left == -1 or right == -1:
        return None
    five = seq[left].decode('ascii').upper()
    three = seq[right].decode('ascii').upper()
    return five, three


# ---------------------------------------------------------------------------
# Downstream-triplet context (5'-[REF>ALT]d1d2-3')
#
# An alternative context that classifies each substitution by the mutated base
# plus its **two downstream** bases (read 5'->3'), rather than the one 5' and one
# 3' flank of the trinucleotide model. The biological motivation is CHG cytosine
# methylation in ascomycete fungi: methyl-C in a ``C-H-G`` context (H = A/C/T)
# deaminates to T, so methylation-driven C>T shows up as an excess of C>T where
# the mutated C is followed by H then G -- a two-base-downstream signal the
# trinucleotide context cannot resolve.
#
# The two downstream bases are always read on the **pyrimidine strand** so the
# counts are invariant to the orientation of the input alignment (see
# :func:`downstream_context`): for a pyrimidine reference the bases are read
# directly downstream, and for a purine reference they are the reverse-complement
# of the two upstream bases (which are the physical downstream bases of the
# equivalent pyrimidine event on the opposite strand).
# ---------------------------------------------------------------------------

# The downstream model folds onto the same six pyrimidine substitution classes as
# SBS-96; only the flank context differs.
DOWNSTREAM_SUBSTITUTIONS: Tuple[Tuple[str, str], ...] = SBS96_SUBSTITUTIONS

# Parses a downstream channel label back into ``(ref, alt, d1, d2)``. The absence
# of a leading flank base (unlike SBS-96's ``N[N>N]N``) makes the two label forms
# unambiguous.
_DOWNSTREAM_LABEL_RE = re.compile(r'^\[([ACGT])>([ACGT])\]([ACGT])([ACGT])$')


def _downstream_label(ref: str, alt: str, d1: str, d2: str) -> str:
    """
    Build a downstream-triplet ``[REF>ALT]d1d2`` channel label.

    Parameters
    ----------
    ref : str
        The reference (mutated) base.
    alt : str
        The derived base.
    d1 : str
        The first downstream base (immediately 3' on the pyrimidine strand).
    d2 : str
        The second downstream base.

    Returns
    -------
    str
        The channel label, e.g. ``[C>T]AG``.
    """
    return f'[{ref}>{alt}]{d1}{d2}'


def _build_downstream_channels() -> List[str]:
    """
    Enumerate downstream-triplet channel labels in canonical order.

    The substitution type varies in the outer loop, the first downstream base in
    the middle loop and the second downstream base in the inner loop, so the
    sixteen ``(d1, d2)`` contexts of each substitution type are contiguous -- the
    same block layout SBS-96 uses for its flanks.

    Returns
    -------
    list of str
        The 96 downstream channel labels.
    """
    channels: List[str] = []
    for ref, alt in DOWNSTREAM_SUBSTITUTIONS:
        for d1 in BASES:
            for d2 in BASES:
                channels.append(_downstream_label(ref, alt, d1, d2))
    return channels


# Canonical downstream channel ordering and its reverse lookup.
DOWNSTREAM_CHANNELS: List[str] = _build_downstream_channels()
DOWNSTREAM_INDEX = {label: i for i, label in enumerate(DOWNSTREAM_CHANNELS)}


def downstream_channel(ref: str, alt: str, d1: str, d2: str) -> str:
    """
    Return the downstream-triplet channel label for a pyrimidine-folded event.

    This is a pure label builder: ``ref``/``alt`` are expected to already be on
    the pyrimidine strand (``ref`` in ``C``/``T``) and ``d1``/``d2`` to already be
    the pyrimidine-strand downstream bases, exactly as :func:`downstream_context`
    resolves them. Folding of a purine-reference event is the caller's
    responsibility (mirroring how :func:`sbs96_channel` is composed via
    :func:`fold_to_pyrimidine`), because the two downstream bases are selected from
    different physical neighbours depending on the reference strand.

    Parameters
    ----------
    ref : str
        The pyrimidine reference base (``C`` or ``T``).
    alt : str
        The derived base on the pyrimidine strand.
    d1 : str
        The first downstream base on the pyrimidine strand.
    d2 : str
        The second downstream base on the pyrimidine strand.

    Returns
    -------
    str
        The channel label, e.g. ``[C>T]AG``.
    """
    return _downstream_label(ref, alt, d1, d2)


def parse_downstream_channel(label: str) -> Tuple[str, str, str, str]:
    """
    Parse a downstream channel label into its ``(ref, alt, d1, d2)`` components.

    Parameters
    ----------
    label : str
        A channel label of the form ``[REF>ALT]d1d2`` (e.g. ``[C>T]AG``).

    Returns
    -------
    tuple of str
        ``(ref, alt, d1, d2)``.

    Raises
    ------
    ValueError
        If ``label`` is not a valid downstream channel label.
    """
    match = _DOWNSTREAM_LABEL_RE.match(label)
    if match is None:
        raise ValueError(f'Not a downstream channel label: {label!r}')
    ref, alt, d1, d2 = match.groups()
    return ref, alt, d1, d2


def downstream_context(
    seq: np.ndarray,
    col: int,
    next_idx: np.ndarray,
    prev_idx: np.ndarray,
) -> Optional[Tuple[str, str]]:
    """
    Resolve the two pyrimidine-strand downstream bases of a column.

    The reference base at ``col`` sets the strand on which "downstream" is read so
    that the result is invariant to the orientation of the input alignment:

    - **Pyrimidine reference (C/T)**: the two downstream bases are read directly
      from the nearest non-gap neighbours to the right (``next_idx`` chained once
      to reach the second base).
    - **Purine reference (A/G)**: the equivalent pyrimidine event lives on the
      opposite strand, whose two downstream bases are the reverse-complement of the
      two **upstream** bases here. So the two nearest non-gap neighbours to the
      left (``prev_idx`` chained once) are complemented and returned.

    Non-ACGT positions must already be normalised to gaps in ``seq`` so the
    neighbour indices skip over them (as for :func:`trinucleotide_context`). When
    either required second neighbour is absent (a terminal column, or -- for a
    purine reference -- a column too close to the 5' end) the event has no full
    downstream context and ``None`` is returned. This 5'/3' asymmetry is inherent
    to reading a fixed two-base window on the pyrimidine strand.

    Parameters
    ----------
    seq : numpy.ndarray
        1-D byte array (dtype ``'S1'``) of the reference/ancestral sequence, with
        every non-``ACGT`` position already normalised to a gap.
    col : int
        The column of the mutated (reference) base.
    next_idx : numpy.ndarray
        1-D int array; the column index of the nearest non-gap base to the right
        of each column, or ``-1``.
    prev_idx : numpy.ndarray
        1-D int array; the column index of the nearest non-gap base to the left of
        each column, or ``-1``.

    Returns
    -------
    tuple of str or None
        ``(d1, d2)`` uppercase pyrimidine-strand downstream bases, or ``None`` if
        the full two-base context is unresolvable.
    """
    ref = seq[col].decode('ascii').upper()
    if ref in PYRIMIDINES:
        first = int(next_idx[col])
        if first == -1:
            return None
        second = int(next_idx[first])
        if second == -1:
            return None
        d1 = seq[first].decode('ascii').upper()
        d2 = seq[second].decode('ascii').upper()
        return d1, d2
    # Purine reference: the pyrimidine-strand downstream bases are the
    # reverse-complement of the two nearest non-gap upstream bases.
    first = int(prev_idx[col])
    if first == -1:
        return None
    second = int(prev_idx[first])
    if second == -1:
        return None
    u1 = seq[first].decode('ascii').upper()
    u2 = seq[second].decode('ascii').upper()
    return COMPLEMENT[u1], COMPLEMENT[u2]
