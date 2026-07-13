"""
Trinucleotide-context substitution spectra (SBS-96 / SBS-192) from an alignment.

This module computes single-base-substitution spectra by comparing every aligned
sequence to an inferred ancestral reference (deRIP2's reconstructed consensus)
and reading each substitution's trinucleotide context from that ancestor. It is
the tree-free *baseline* method: every difference between a tip and the single
reference is counted as one event, with its 5'/3' context taken from the nearest
non-gap ancestral bases.

Because there is no phylogeny, recurrence can only be reported as a *multi-hit
column* proxy: how many sequences independently carry each derived state at a
site. True independent-event counting requires ancestral reconstruction on a tree
and is provided by the phylogenetic path in :mod:`derip2.spectra` (later
milestone). The API here is deliberately event-stream shaped so that path can
reuse the same channel assembly.

See Also
--------
derip2.spectra.channels : SBS-96/192 channel ordering and pyrimidine folding.
derip2.stats.strand_bias : The RSI statistic, the sibling per-alignment measure.
"""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from derip2.aln_ops import _nongap_neighbors
from derip2.spectra.channels import (
    SBS96_CHANNELS,
    SBS96_INDEX,
    SBS192_CHANNELS,
    SBS192_INDEX,
    sbs96_channel,
    sbs192_channel,
    trinucleotide_context,
)

logger = logging.getLogger(__name__)

# Byte value -> base-code (A,C,G,T -> 0..3), everything else -> -1. Indexed by
# the raw byte so a 256-entry lookup vector resolves an 'S1' array in one pass.
_CODE_LUT = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate((b'A', b'C', b'G', b'T')):
    _CODE_LUT[_b[0]] = _i
    _CODE_LUT[_b.lower()[0]] = _i
_CODE_TO_BASE = np.array([b'A', b'C', b'G', b'T'], dtype='S1')


def _build_channel_tables() -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute the SBS-96 and SBS-192 channel index for every base-code 4-tuple.

    The tables are indexed ``[five, ref, alt, three]`` by base code (0..3), so an
    event stream expressed as code arrays resolves to channel rows in a single
    vectorised gather. Cells where ``ref == alt`` (not a substitution) hold ``-1``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(idx96, idx192)``, each a ``(4, 4, 4, 4)`` int64 array.
    """
    idx96 = np.full((4, 4, 4, 4), -1, dtype=np.int64)
    idx192 = np.full((4, 4, 4, 4), -1, dtype=np.int64)
    for fc in range(4):
        five = _CODE_TO_BASE[fc].decode('ascii')
        for rc in range(4):
            ref = _CODE_TO_BASE[rc].decode('ascii')
            for ac in range(4):
                if ac == rc:
                    continue
                alt = _CODE_TO_BASE[ac].decode('ascii')
                for tc in range(4):
                    three = _CODE_TO_BASE[tc].decode('ascii')
                    idx96[fc, rc, ac, tc] = SBS96_INDEX[
                        sbs96_channel(five, ref, alt, three)
                    ]
                    idx192[fc, rc, ac, tc] = SBS192_INDEX[
                        sbs192_channel(five, ref, alt, three)
                    ]
    return idx96, idx192


# Shared channel-index lookup tables, used by every spectrum-assembly path.
IDX96_TABLE, IDX192_TABLE = _build_channel_tables()


def assemble_matrices(
    five_c: np.ndarray,
    ref_c: np.ndarray,
    alt_c: np.ndarray,
    three_c: np.ndarray,
    sample_c: np.ndarray,
    n_samples: int,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulate an event stream into SBS-96 and SBS-192 count matrices.

    This is the single assembly core shared by the tree-free baseline and the
    phylogenetic branch-traversal path. Each event is a tuple of base codes
    (``five``, ``ref``, ``alt``, ``three``) plus the sample it belongs to; the
    precomputed :data:`IDX96_TABLE` / :data:`IDX192_TABLE` map it to its two
    channel rows.

    Parameters
    ----------
    five_c, ref_c, alt_c, three_c : numpy.ndarray
        ``(n_events,)`` int arrays of base codes (0..3) for the 5' flank,
        reference, derived and 3' flank of each event.
    sample_c : numpy.ndarray
        ``(n_events,)`` int array indexing the sample columns.
    n_samples : int
        Number of sample columns in the output matrices.
    weights : numpy.ndarray, optional
        ``(n_events,)`` float weights (e.g. posterior products). Default ``None``
        counts every event as 1.

    Returns
    -------
    tuple of numpy.ndarray
        ``(sbs96, sbs192)`` of shapes ``(96, n_samples)`` and ``(192, n_samples)``.
    """
    sbs96 = np.zeros((96, n_samples), dtype=np.float64)
    sbs192 = np.zeros((192, n_samples), dtype=np.float64)
    if five_c.size:
        ch96 = IDX96_TABLE[five_c, ref_c, alt_c, three_c]
        ch192 = IDX192_TABLE[five_c, ref_c, alt_c, three_c]
        w = 1.0 if weights is None else weights
        np.add.at(sbs96, (ch96, sample_c), w)
        np.add.at(sbs192, (ch192, sample_c), w)
    return sbs96, sbs192


@dataclass(frozen=True)
class SpectraResult:
    """
    Trinucleotide-context substitution spectra for an alignment.

    Attributes
    ----------
    sbs96 : numpy.ndarray
        ``(96, n_samples)`` float array of SBS-96 (pyrimidine-collapsed) counts,
        in the canonical channel order of
        :data:`derip2.spectra.channels.SBS96_CHANNELS`.
    sbs192 : numpy.ndarray
        ``(192, n_samples)`` float array of strand-resolved SBS-192 counts, in
        the order of :data:`derip2.spectra.channels.SBS192_CHANNELS`.
    sample_names : list of str
        Column labels for the two matrices, one per sample.
    event_rows, event_cols : numpy.ndarray
        ``(n_events,)`` int arrays: the alignment row and column of every counted
        substitution, in discovery order.
    event_ref, event_alt : numpy.ndarray
        ``(n_events,)`` ``'S1'`` arrays: the reference (ancestral) and derived
        base of every event.
    event_five, event_three : numpy.ndarray
        ``(n_events,)`` ``'S1'`` arrays: the 5' and 3' flanking bases read from
        the ancestor.
    event_sample : numpy.ndarray
        ``(n_events,)`` int array indexing :attr:`sample_names`.
    homoplasy_counts : numpy.ndarray
        ``(n_cols, 4)`` int array: for each column, how many sequences carry each
        derived base (A, C, G, T) as a substitution away from the ancestor. A
        column with a value >= 2 was hit independently more than once (baseline
        proxy for recurrence).
    ancestor_ref : numpy.ndarray
        ``(n_cols,)`` ``'S1'`` array of the reference base per column. For the
        baseline this is the single ancestor; for the phylogenetic path it is the
        reconstructed root sequence.
    n_indel_or_ambiguous : int
        Number of tip/ancestor differences skipped because one side was a gap or
        a non-ACGT base (not a callable substitution).
    n_unassignable_context : int
        Number of substitutions dropped because a full trinucleotide context
        could not be resolved (terminal columns).
    method : str
        ``'baseline'`` for the single-reference spectra or ``'phylogenetic'`` for
        the branch-traversal spectra.
    event_parent_names, event_child_names : list of str or None
        For the phylogenetic path, the parent and child node names of every
        event's edge (``None`` for the baseline).
    """

    sbs96: np.ndarray
    sbs192: np.ndarray
    sample_names: List[str]
    event_rows: np.ndarray
    event_cols: np.ndarray
    event_ref: np.ndarray
    event_alt: np.ndarray
    event_five: np.ndarray
    event_three: np.ndarray
    event_sample: np.ndarray
    homoplasy_counts: np.ndarray
    ancestor_ref: np.ndarray
    n_indel_or_ambiguous: int
    n_unassignable_context: int
    method: str = 'baseline'
    event_parent_names: Optional[List[str]] = None
    event_child_names: Optional[List[str]] = None

    @property
    def sbs96_channels(self) -> List[str]:
        """
        Canonical SBS-96 channel labels, aligned to :attr:`sbs96` rows.

        Returns
        -------
        list of str
            The 96 channel labels in row order.
        """
        return list(SBS96_CHANNELS)

    @property
    def sbs192_channels(self) -> List[str]:
        """
        Canonical SBS-192 channel labels, aligned to :attr:`sbs192` rows.

        Returns
        -------
        list of str
            The 192 channel labels in row order.
        """
        return list(SBS192_CHANNELS)

    def event_records(self) -> List[Dict]:
        """
        Return every counted substitution as a list of dictionaries.

        Returns
        -------
        list of dict
            One dictionary per event, in discovery order, with keys ``sample``,
            ``row``, ``col``, ``ref``, ``alt``, ``five_prime``, ``three_prime``,
            ``sbs96`` and ``sbs192`` (the two channel labels). Phylogenetic
            results additionally carry ``parent`` and ``child`` node names, and
            their ``row`` is the child-node ordinal rather than an alignment row.
        """
        records: List[Dict] = []
        for i in range(self.event_rows.size):
            five = self.event_five[i].decode('ascii')
            three = self.event_three[i].decode('ascii')
            ref = self.event_ref[i].decode('ascii')
            alt = self.event_alt[i].decode('ascii')
            record = {
                'sample': self.sample_names[int(self.event_sample[i])],
                'row': int(self.event_rows[i]),
                'col': int(self.event_cols[i]),
                'ref': ref,
                'alt': alt,
                'five_prime': five,
                'three_prime': three,
                'sbs96': sbs96_channel(five, ref, alt, three),
                'sbs192': sbs192_channel(five, ref, alt, three),
            }
            if self.event_child_names is not None:
                record['parent'] = self.event_parent_names[i]
                record['child'] = self.event_child_names[i]
            records.append(record)
        return records

    def homoplasy_table(self, min_hits: int = 2) -> List[Dict]:
        """
        Return columns hit by the same substitution in >= ``min_hits`` sequences.

        Parameters
        ----------
        min_hits : int, optional
            Minimum number of independent sequences carrying the same derived
            base at a column for it to be reported (default: 2).

        Returns
        -------
        list of dict
            One dictionary per (column, derived base) meeting the threshold, with
            keys ``col``, ``ref``, ``alt``, ``n_independent``, sorted by
            descending ``n_independent`` then column.
        """
        rows: List[Dict] = []
        cols, alts = np.nonzero(self.homoplasy_counts >= min_hits)
        for col, alt_code in zip(cols, alts):
            ref = self.ancestor_ref[col].decode('ascii')
            rows.append(
                {
                    'col': int(col),
                    'ref': ref,
                    'alt': _CODE_TO_BASE[alt_code].decode('ascii'),
                    'n_independent': int(self.homoplasy_counts[col, alt_code]),
                }
            )
        rows.sort(key=lambda r: (-r['n_independent'], r['col']))
        return rows

    def as_dict(self) -> Dict:
        """
        Return a JSON-serialisable summary of the spectra.

        Used for golden regression tests. Event-level detail is omitted; the
        matrices, sample names, homoplasy hits and skip counts fully pin the
        result.

        Returns
        -------
        dict
            Nested dictionary of the SBS-96/192 matrices (as nested lists), the
            sample names, the >= 2 homoplasy table and the two skip counts.
        """
        return {
            'method': self.method,
            'sample_names': list(self.sample_names),
            'sbs96': self.sbs96.tolist(),
            'sbs192': self.sbs192.tolist(),
            'homoplasy': self.homoplasy_table(min_hits=2),
            'n_indel_or_ambiguous': int(self.n_indel_or_ambiguous),
            'n_unassignable_context': int(self.n_unassignable_context),
        }


def _sanitize_ancestor(ancestor_seq: str, n_cols: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the ancestral base array and its nearest non-gap neighbour indices.

    Non-``ACGT`` positions (gaps, ``N``, IUPAC codes) are treated as gaps so that
    both the reference-base test and the flank lookup skip over them, exactly as
    the nearest-non-gap context rule requires.

    Parameters
    ----------
    ancestor_seq : str
        The gapped ancestral/consensus sequence.
    n_cols : int
        Expected alignment length; validated against ``ancestor_seq``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(anc_bytes, code)`` where ``anc_bytes`` is a sanitised ``(n_cols,)``
        ``'S1'`` array (non-ACGT -> ``b'-'``) and ``code`` is a ``(n_cols,)`` int
        array of base codes (``-1`` for gap/ambiguous).

    Raises
    ------
    ValueError
        If ``ancestor_seq`` length does not match ``n_cols``.
    """
    if len(ancestor_seq) != n_cols:
        raise ValueError(
            f'Ancestor length {len(ancestor_seq)} does not match alignment '
            f'width {n_cols}'
        )
    raw = np.frombuffer(ancestor_seq.upper().encode('ascii'), dtype='S1').copy()
    code = _CODE_LUT[raw.view(np.uint8)]
    anc_bytes = np.where(code >= 0, raw, b'-')
    return anc_bytes, code


def _resolve_samples(
    samples: Optional[Sequence[str]], n_rows: int
) -> Tuple[List[str], np.ndarray]:
    """
    Map an optional per-row sample labelling to sample names and row indices.

    Parameters
    ----------
    samples : sequence of str or None
        Per-row sample label (length ``n_rows``), or ``None`` for a single pooled
        sample named ``AllSequences``.
    n_rows : int
        Number of alignment rows.

    Returns
    -------
    tuple
        ``(sample_names, row_to_sample)`` where ``sample_names`` is the ordered
        unique labels and ``row_to_sample`` is a ``(n_rows,)`` int array indexing
        them.

    Raises
    ------
    ValueError
        If ``samples`` is given but its length does not match ``n_rows``.
    """
    if samples is None:
        return ['AllSequences'], np.zeros(n_rows, dtype=np.int64)
    if len(samples) != n_rows:
        raise ValueError(
            f'samples has length {len(samples)} but the alignment has {n_rows} rows'
        )
    names: List[str] = []
    index: Dict[str, int] = {}
    row_to_sample = np.empty(n_rows, dtype=np.int64)
    for r, label in enumerate(samples):
        if label not in index:
            index[label] = len(names)
            names.append(label)
        row_to_sample[r] = index[label]
    return names, row_to_sample


def compute_spectra(
    column_classes,
    ancestor_seq: str,
    *,
    samples: Optional[Sequence[str]] = None,
) -> SpectraResult:
    """
    Compute SBS-96 and SBS-192 spectra against a single ancestral reference.

    Every alignment cell whose base differs from the ancestral base at that
    column, where both are unambiguous (``ACGT``), is counted as one substitution
    event. The 5'/3' context is read from the ancestor using the nearest non-gap
    bases, matching the Alexandrov reference-context convention. Events are folded
    to the pyrimidine strand for SBS-96 and kept strand-resolved for SBS-192.

    Parameters
    ----------
    column_classes : derip2.aln_ops.ColumnClassification
        The cached per-cell classification; only its ``arr`` (the ``(n_rows,
        n_cols)`` observed-base byte array) is consumed here, guaranteeing the
        spectra are computed over the same alignment the correction used.
    ancestor_seq : str
        The gapped ancestral/consensus sequence, one base per alignment column
        (deRIP2's ``gapped_consensus``).
    samples : sequence of str or None, optional
        Per-row sample label to split the matrices by (length ``n_rows``).
        Default ``None`` pools every sequence into one ``AllSequences`` sample.

    Returns
    -------
    SpectraResult
        The assembled spectra, per-event detail, homoplasy proxy and skip counts.
    """
    arr = column_classes.arr
    n_rows, n_cols = arr.shape
    sample_names, row_to_sample = _resolve_samples(samples, n_rows)
    n_samples = len(sample_names)

    anc_bytes, anc_code = _sanitize_ancestor(ancestor_seq, n_cols)
    next_idx_2d, prev_idx_2d = _nongap_neighbors(anc_bytes.reshape(1, n_cols))
    next_idx = next_idx_2d[0]
    prev_idx = prev_idx_2d[0]

    # Observed-base codes for the whole alignment in one vectorised pass.
    obs_code = _CODE_LUT[arr.view(np.uint8)].reshape(n_rows, n_cols)

    # A callable substitution needs an unambiguous ancestor and observed base
    # that differ. Anything else is an indel/ambiguous non-substitution.
    anc_code_row = anc_code.reshape(1, n_cols)
    both_valid = (obs_code >= 0) & (anc_code_row >= 0)
    is_diff = obs_code != anc_code_row
    substitution = both_valid & is_diff
    n_indel_or_ambiguous = int((is_diff & ~both_valid).sum())

    # Per-column context: resolve the 5'/3' flank codes once per column using the
    # nearest non-gap ancestral bases. Columns with a valid ancestral base but no
    # resolvable context are flagged so their substitutions can be reported as
    # unassignable rather than silently dropped.
    five_code_per_col = np.full(n_cols, -1, dtype=np.int64)
    three_code_per_col = np.full(n_cols, -1, dtype=np.int64)
    context_ok = np.zeros(n_cols, dtype=bool)
    for col in range(n_cols):
        if anc_code[col] < 0:
            continue
        ctx = trinucleotide_context(anc_bytes, col, next_idx, prev_idx)
        if ctx is None:
            continue
        five, three = ctx
        five_code_per_col[col] = _CODE_LUT[ord(five)]
        three_code_per_col[col] = _CODE_LUT[ord(three)]
        context_ok[col] = True

    # Substitutions in columns whose context could not be resolved are counted
    # but excluded from the matrices.
    ctx_missing = substitution & ~context_ok.reshape(1, n_cols)
    n_unassignable_context = int(ctx_missing.sum())
    countable = substitution & context_ok.reshape(1, n_cols)

    rows, cols = np.nonzero(countable)
    alt_codes = obs_code[rows, cols].astype(np.int64)
    ref_codes = anc_code[cols].astype(np.int64)
    five_codes = five_code_per_col[cols]
    three_codes = three_code_per_col[cols]
    ev_sample = row_to_sample[rows]

    sbs96, sbs192 = assemble_matrices(
        five_codes, ref_codes, alt_codes, three_codes, ev_sample, n_samples
    )

    # Homoplasy proxy: count, per column, how many sequences independently carry
    # each derived base as a substitution (over all countable substitutions).
    homoplasy_counts = np.zeros((n_cols, 4), dtype=np.int64)
    if rows.size:
        np.add.at(homoplasy_counts, (cols, alt_codes), 1)

    event_ref = anc_bytes[cols]
    event_alt = _CODE_TO_BASE[alt_codes]
    event_five = _CODE_TO_BASE[five_codes]
    event_three = _CODE_TO_BASE[three_codes]

    logger.debug(
        'compute_spectra: %d events across %d samples '
        '(%d indel/ambiguous, %d unassignable context)',
        rows.size,
        n_samples,
        n_indel_or_ambiguous,
        n_unassignable_context,
    )

    return SpectraResult(
        sbs96=sbs96,
        sbs192=sbs192,
        sample_names=sample_names,
        event_rows=rows,
        event_cols=cols,
        event_ref=event_ref,
        event_alt=event_alt,
        event_five=event_five,
        event_three=event_three,
        event_sample=ev_sample,
        homoplasy_counts=homoplasy_counts,
        ancestor_ref=anc_bytes,
        n_indel_or_ambiguous=n_indel_or_ambiguous,
        n_unassignable_context=n_unassignable_context,
    )
