"""
Flanking-context spectra of RIP-like sites from an alignment.

This module answers a specific biological question: among the RIP substrate
dinucleotides (``CpA``, or ``TpG`` when the ``CpA`` is on the reverse strand) that
*survive* in otherwise RIP-affected sequences, is there a local sequence context
that protects them from deamination? To probe it we classify every RIP-like
dinucleotide by the single base one position upstream and one downstream — a 4 bp
motif ``[up][center][down]`` — and compare the flank-context distribution of
surviving **substrate** sites against that of realised **product** (``TpA``) sites.

Two site states are counted per sequence, using the boolean cell masks of a
:class:`derip2.aln_ops.ColumnClassification` (the single source of truth for RIP
context, gap-aware over nearest non-gap neighbours):

- **Substrate** — ``cls.ca`` (``C`` followed by ``A``) and ``cls.tg`` (``G``
  preceded by ``T``), counted **anywhere** in the sequence, not only in
  RIP-informative columns. This deliberately uses the raw dinucleotide masks, not
  the ``ct_ok``/``ga_ok``-gated ``sub_fwd``/``sub_rev`` used by the strand-bias
  statistic, because the question is about every surviving substrate.
- **Product** — ``cls.prod_fwd`` (``ta & fwd_col``) and ``cls.prod_rev``
  (``ta2 & rev_col``): a ``TpA`` sitting in a column that also shows a surviving
  substrate, so the product is attributable to RIP.

Each state yields a ``(16, n_rows)`` count matrix (one column per alignment row),
folded so every reverse-strand motif is reverse-complemented onto the
``CA``/``TA``-equivalent channel (see :mod:`derip2.spectra.flank_channels`).

See Also
--------
derip2.spectra.flank_channels : The 16-channel labelling and fold lookup.
derip2.stats.spectra_compare : The cosine / chi-squared comparison reused here.
derip2.stats.mutation_spectra : The sibling trinucleotide SBS-96/192 spectra.
"""

from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from derip2.spectra.flank_channels import (
    COMP_CODE,
    FLANK16_LABELS_CA,
    FLANK16_LABELS_TA,
    FLANK16_PAIR_LABELS,
    IDX16_TABLE,
)
from derip2.stats.mutation_spectra import _CODE_LUT
from derip2.stats.spectra_compare import compare_spectra

logger = logging.getLogger(__name__)

# The four raw mask-derived count states, in a fixed order for serialisation.
STATE_KEYS: Tuple[str, ...] = ('sub_fwd', 'sub_rev', 'prod_fwd', 'prod_rev')

# The five per-sequence comparisons produced by :func:`compare_flank_spectra`.
COMPARISON_KEYS: Tuple[str, ...] = (
    'sub_vs_prod_combined',
    'sub_vs_prod_fwd',
    'sub_vs_prod_rev',
    'fwd_vs_rev_substrate',
    'fwd_vs_rev_product',
)


@dataclass(frozen=True)
class FlankSpectraResult:
    """
    Per-sequence 16-channel flanking-context spectra of RIP-like sites.

    Four count matrices, each ``(16, n_rows)`` with one column per alignment row
    (the row *is* the sample index). Every motif is folded so its centre is
    ``CA`` (substrate) or ``TA`` (product) and its channel is indexed by the two
    resolved flanks ``up*4 + down`` (see :mod:`derip2.spectra.flank_channels`).

    Attributes
    ----------
    sub_fwd, sub_rev : numpy.ndarray
        ``(16, n_rows)`` float counts of forward (``CpA``) and reverse (``TpG``)
        substrate sites, counted anywhere in each sequence.
    prod_fwd, prod_rev : numpy.ndarray
        ``(16, n_rows)`` float counts of forward and reverse RIP product
        (``TpA``) sites in RIP-informative columns.
    sample_names : list of str
        Column labels, one per alignment row.
    n_skipped_flank : dict of str to int
        Per-state count of sites dropped because an up or down flank could not be
        resolved to an ``ACGT`` base (terminal columns, or a non-ACGT neighbour).
        Keyed by :data:`STATE_KEYS`.
    channels_substrate, channels_product : list of str
        The 16 four-base motif labels for the substrate (``CA``) and product
        (``TA``) states, aligned to the matrix rows.
    """

    sub_fwd: np.ndarray
    sub_rev: np.ndarray
    prod_fwd: np.ndarray
    prod_rev: np.ndarray
    sample_names: List[str]
    n_skipped_flank: Dict[str, int]
    channels_substrate: List[str] = field(
        default_factory=lambda: list(FLANK16_LABELS_CA)
    )
    channels_product: List[str] = field(default_factory=lambda: list(FLANK16_LABELS_TA))

    def substrate_combined(self) -> np.ndarray:
        """
        Combined-strand substrate spectrum.

        Returns
        -------
        numpy.ndarray
            ``(16, n_rows)`` sum of :attr:`sub_fwd` and :attr:`sub_rev`.
        """
        return self.sub_fwd + self.sub_rev

    def product_combined(self) -> np.ndarray:
        """
        Combined-strand product spectrum.

        Returns
        -------
        numpy.ndarray
            ``(16, n_rows)`` sum of :attr:`prod_fwd` and :attr:`prod_rev`.
        """
        return self.prod_fwd + self.prod_rev

    def matrix(self, state: str, strand: str) -> np.ndarray:
        """
        Return one ``(16, n_rows)`` count matrix by state and strand.

        Parameters
        ----------
        state : {'substrate', 'product'}
            Which site state to return.
        strand : {'combined', 'forward', 'reverse'}
            Which strand's counts (``'combined'`` sums the two strands).

        Returns
        -------
        numpy.ndarray
            The ``(16, n_rows)`` matrix.

        Raises
        ------
        ValueError
            If ``state`` or ``strand`` is not recognised.
        """
        if state == 'substrate':
            fwd, rev = self.sub_fwd, self.sub_rev
        elif state == 'product':
            fwd, rev = self.prod_fwd, self.prod_rev
        else:
            raise ValueError(f"state must be 'substrate' or 'product', got {state!r}")
        if strand == 'forward':
            return fwd
        if strand == 'reverse':
            return rev
        if strand == 'combined':
            return fwd + rev
        raise ValueError(
            f"strand must be 'combined', 'forward' or 'reverse', got {strand!r}"
        )

    def pooled(self) -> Dict[str, np.ndarray]:
        """
        Pool every sequence into a single alignment-wide spectrum per matrix.

        Returns
        -------
        dict of str to numpy.ndarray
            Keys :data:`STATE_KEYS`; each value a ``(16,)`` row-summed vector.
        """
        return {
            'sub_fwd': self.sub_fwd.sum(axis=1),
            'sub_rev': self.sub_rev.sum(axis=1),
            'prod_fwd': self.prod_fwd.sum(axis=1),
            'prod_rev': self.prod_rev.sum(axis=1),
        }

    def as_dict(self) -> Dict:
        """
        Return a JSON-serialisable summary for golden regression tests.

        Returns
        -------
        dict
            The four count matrices as nested lists, the sample names, the
            per-state skipped-flank counts and the two channel-label sets.
        """
        return {
            'sample_names': list(self.sample_names),
            'channels_substrate': list(self.channels_substrate),
            'channels_product': list(self.channels_product),
            'sub_fwd': self.sub_fwd.tolist(),
            'sub_rev': self.sub_rev.tolist(),
            'prod_fwd': self.prod_fwd.tolist(),
            'prod_rev': self.prod_rev.tolist(),
            'n_skipped_flank': {k: int(v) for k, v in self.n_skipped_flank.items()},
        }


def _codes_at(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """
    Look up base codes (A,C,G,T -> 0..3, else -1) at given cells of a byte array.

    Parameters
    ----------
    arr : numpy.ndarray
        ``(n_rows, n_cols)`` ``'S1'`` alignment byte array.
    rows, cols : numpy.ndarray
        Equal-length integer index arrays of the cells to read. ``cols`` may
        contain ``-1`` for a missing neighbour; those entries return ``-1`` (the
        code for a non-ACGT base) rather than wrapping to the last column.

    Returns
    -------
    numpy.ndarray
        Integer base codes, ``-1`` where the base is absent or non-ACGT.
    """
    codes = np.full(cols.shape, -1, dtype=np.int64)
    present = cols >= 0
    if present.any():
        bytes_here = arr[rows[present], cols[present]].view(np.uint8)
        codes[present] = _CODE_LUT[bytes_here]
    return codes


def _gather_left_anchored(
    mask: np.ndarray,
    arr: np.ndarray,
    next_idx: np.ndarray,
    prev_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Gather flank codes for a mask anchored on the *left* centre base.

    Used for the forward-strand masks ``ca`` (``C`` at the cell, ``A`` at the next
    non-gap column) and ``ta``. The 4 bp motif is ``[up][C][A][down]`` on the
    forward strand, so the flanks need no folding.

    Parameters
    ----------
    mask : numpy.ndarray
        ``(n_rows, n_cols)`` boolean cell mask; ``True`` at the left centre base.
    arr : numpy.ndarray
        ``(n_rows, n_cols)`` ``'S1'`` alignment byte array.
    next_idx, prev_idx : numpy.ndarray
        ``(n_rows, n_cols)`` nearest non-gap neighbour indices (``-1`` if none).

    Returns
    -------
    tuple
        ``(rows, up_code, down_code, n_skipped)`` for sites whose up and down
        flanks both resolve to an ``ACGT`` base; ``n_skipped`` counts the rest.
    """
    rows, cols = np.where(mask)
    up_idx = prev_idx[rows, cols]
    mid_idx = next_idx[rows, cols]  # the A of the centre dinucleotide
    down_idx = next_idx[rows, mid_idx]  # one base past the centre
    # A missing centre partner (mid_idx == -1) would make down_idx meaningless;
    # guard it so the chained lookup never wraps.
    down_idx = np.where(mid_idx >= 0, down_idx, -1)

    up_code = _codes_at(arr, rows, up_idx)
    down_code = _codes_at(arr, rows, down_idx)
    valid = (up_code >= 0) & (down_code >= 0)
    n_skipped = int((~valid).sum())
    return rows[valid], up_code[valid], down_code[valid], n_skipped


def _gather_right_anchored(
    mask: np.ndarray,
    arr: np.ndarray,
    next_idx: np.ndarray,
    prev_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Gather folded flank codes for a mask anchored on the *right* centre base.

    Used for the reverse-strand masks ``tg`` (``G`` at the cell, ``T`` at the
    previous non-gap column) and ``ta2``. The physical forward-strand motif is
    ``[up][T][X][down]``; reverse-complementing it to the ``CA``/``TA``-equivalent
    swaps the flanks and complements them, so the folded upstream base is the
    complement of the physical *downstream* base and vice versa.

    Parameters
    ----------
    mask : numpy.ndarray
        ``(n_rows, n_cols)`` boolean cell mask; ``True`` at the right centre base.
    arr : numpy.ndarray
        ``(n_rows, n_cols)`` ``'S1'`` alignment byte array.
    next_idx, prev_idx : numpy.ndarray
        ``(n_rows, n_cols)`` nearest non-gap neighbour indices (``-1`` if none).

    Returns
    -------
    tuple
        ``(rows, up_code, down_code, n_skipped)`` where ``up_code``/``down_code``
        are already folded onto the ``CA``/``TA`` strand.
    """
    rows, cols = np.where(mask)
    mid_idx = prev_idx[rows, cols]  # the T (left centre base)
    up_idx = prev_idx[rows, mid_idx]  # physical upstream flank, left of the T
    down_idx = next_idx[rows, cols]  # physical downstream flank, right of anchor
    # Guard the chained upstream lookup against a missing left centre base.
    up_idx = np.where(mid_idx >= 0, up_idx, -1)

    phys_up = _codes_at(arr, rows, up_idx)
    phys_down = _codes_at(arr, rows, down_idx)
    valid = (phys_up >= 0) & (phys_down >= 0)
    n_skipped = int((~valid).sum())
    # Fold (reverse-complement): swap sides and complement each flank.
    up_code = COMP_CODE[phys_down[valid]]
    down_code = COMP_CODE[phys_up[valid]]
    return rows[valid], up_code, down_code, n_skipped


def _assemble16(
    rows: np.ndarray, up_code: np.ndarray, down_code: np.ndarray, n_rows: int
) -> np.ndarray:
    """
    Scatter resolved flank pairs into a ``(16, n_rows)`` count matrix.

    Parameters
    ----------
    rows : numpy.ndarray
        Alignment row (sample index) of every counted site.
    up_code, down_code : numpy.ndarray
        Folded upstream and downstream flank base codes (0..3) of every site.
    n_rows : int
        Number of alignment rows (matrix columns).

    Returns
    -------
    numpy.ndarray
        ``(16, n_rows)`` float count matrix.
    """
    out = np.zeros((16, n_rows), dtype=np.float64)
    if rows.size:
        channel = IDX16_TABLE[up_code, down_code]
        np.add.at(out, (channel, rows), 1.0)
    return out


def compute_flank_spectra(
    column_classes, sample_names: Optional[List[str]] = None
) -> FlankSpectraResult:
    """
    Compute per-sequence 16-channel flanking-context spectra of RIP-like sites.

    Substrate sites (``cls.ca`` / ``cls.tg``) are counted anywhere in each
    sequence; product sites (``cls.prod_fwd`` / ``cls.prod_rev``) only in
    RIP-informative columns. Each site's 4 bp motif is resolved over the nearest
    non-gap neighbours and folded onto the ``CA``/``TA``-equivalent channel. The
    computation is fully vectorised over the whole alignment; the alignment row is
    the sample index, so no per-row Python loop is needed.

    Parameters
    ----------
    column_classes : derip2.aln_ops.ColumnClassification
        The RIP classification, providing ``arr``, ``next_idx``, ``prev_idx`` and
        the ``ca``/``tg`` masks and ``prod_fwd``/``prod_rev`` cell properties.
    sample_names : list of str, optional
        Per-row labels (length ``n_rows``). Defaults to the row ordinals as
        strings.

    Returns
    -------
    FlankSpectraResult
        The four ``(16, n_rows)`` count matrices and per-state skipped counts.

    Raises
    ------
    ValueError
        If ``sample_names`` is given and its length is not ``n_rows``.
    """
    cls = column_classes
    arr = cls.arr
    next_idx = cls.next_idx
    prev_idx = cls.prev_idx
    n_rows = arr.shape[0]

    if sample_names is None:
        sample_names = [str(i) for i in range(n_rows)]
    elif len(sample_names) != n_rows:
        raise ValueError(
            f'sample_names length {len(sample_names)} does not match '
            f'{n_rows} alignment rows'
        )

    skipped: Dict[str, int] = {}
    matrices: Dict[str, np.ndarray] = {}

    # Left-anchored forward masks: no fold. Right-anchored reverse masks: revcomp.
    for key, mask, gather in (
        ('sub_fwd', cls.ca, _gather_left_anchored),
        ('sub_rev', cls.tg, _gather_right_anchored),
        ('prod_fwd', cls.prod_fwd, _gather_left_anchored),
        ('prod_rev', cls.prod_rev, _gather_right_anchored),
    ):
        rows, up_code, down_code, n_skipped = gather(mask, arr, next_idx, prev_idx)
        matrices[key] = _assemble16(rows, up_code, down_code, n_rows)
        skipped[key] = n_skipped

    logger.info(
        'Flank-context spectra: substrate %d (fwd) / %d (rev), product %d / %d '
        'sites across %d sequence(s); %d sites skipped for unresolved flanks',
        int(matrices['sub_fwd'].sum()),
        int(matrices['sub_rev'].sum()),
        int(matrices['prod_fwd'].sum()),
        int(matrices['prod_rev'].sum()),
        n_rows,
        sum(skipped.values()),
    )

    return FlankSpectraResult(
        sub_fwd=matrices['sub_fwd'],
        sub_rev=matrices['sub_rev'],
        prod_fwd=matrices['prod_fwd'],
        prod_rev=matrices['prod_rev'],
        sample_names=list(sample_names),
        n_skipped_flank=skipped,
        channels_substrate=list(FLANK16_LABELS_CA),
        channels_product=list(FLANK16_LABELS_TA),
    )


def compare_flank_spectra(
    result: FlankSpectraResult,
    row_index: int,
    *,
    min_sites: int = 20,
    top: int = 6,
) -> Dict[str, Dict]:
    """
    Run the five per-sequence flank-context comparisons for one alignment row.

    Substrate (``CA``-centred) channel *k* and product (``TA``-centred) channel
    *k* share the same ``(up, down)`` flank context, so the two spectra are
    compared position-by-position as a like-for-like flank-context test (labelled
    by the centre-agnostic ``up.down`` pair). The chi-squared p-value is only
    trustworthy when both spectra carry enough sites; a ``chi2_reliable`` flag
    records whether both totals reach ``min_sites`` so callers can lead with the
    scale-free cosine effect size.

    Parameters
    ----------
    result : FlankSpectraResult
        The computed spectra.
    row_index : int
        Which alignment row (sample column) to compare.
    min_sites : int, optional
        Minimum site count on *both* sides for the chi-squared test to be flagged
        reliable (default: 20).
    top : int, optional
        Number of most-differentiating flank channels to report per comparison
        (default: 6).

    Returns
    -------
    dict of str to dict
        Keyed by :data:`COMPARISON_KEYS`. Each value is the
        :func:`derip2.stats.spectra_compare.compare_spectra` result augmented with
        ``n_a``, ``n_b`` (the two site totals) and ``chi2_reliable``.
    """
    sub_f = result.sub_fwd[:, row_index]
    sub_r = result.sub_rev[:, row_index]
    prod_f = result.prod_fwd[:, row_index]
    prod_r = result.prod_rev[:, row_index]
    sub_c = sub_f + sub_r
    prod_c = prod_f + prod_r

    pairs = {
        'sub_vs_prod_combined': (sub_c, prod_c),
        'sub_vs_prod_fwd': (sub_f, prod_f),
        'sub_vs_prod_rev': (sub_r, prod_r),
        'fwd_vs_rev_substrate': (sub_f, sub_r),
        'fwd_vs_rev_product': (prod_f, prod_r),
    }

    out: Dict[str, Dict] = {}
    for name, (a, b) in pairs.items():
        comp = compare_spectra(a, b, channels=FLANK16_PAIR_LABELS, top=top)
        n_a = float(a.sum())
        n_b = float(b.sum())
        comp['n_a'] = n_a
        comp['n_b'] = n_b
        comp['chi2_reliable'] = bool(n_a >= min_sites and n_b >= min_sites)
        out[name] = comp
    return out
