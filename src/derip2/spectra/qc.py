"""
Alignment quality control for the phylogenetic mutation-spectrum pipeline.

Alignment artefacts become phantom substitutions, so before any tree is built the
alignment is profiled: per-column gap and ambiguity fractions are computed and
columns too gappy to give reliable flanking context are flagged. The report is
advisory — nothing is removed — but the flags let downstream steps and the reader
judge how much of the spectrum rests on well-supported columns.
"""

from dataclasses import dataclass
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Non-ACGT, non-gap symbols are treated as ambiguous.
_ACGT = frozenset(b'ACGTacgt')


@dataclass
class ColumnProfile:
    """
    Per-column gap and ambiguity profile of an alignment.

    Attributes
    ----------
    n_rows : int
        Number of sequences.
    n_cols : int
        Number of alignment columns.
    gap_fraction : numpy.ndarray
        ``(n_cols,)`` fraction of rows that are a gap in each column.
    ambiguous_fraction : numpy.ndarray
        ``(n_cols,)`` fraction of rows that are a non-ACGT, non-gap symbol.
    context_unreliable : numpy.ndarray
        ``(n_cols,)`` boolean; True where the gap fraction exceeds the threshold,
        so flanking context resolved through the column should be distrusted.
    gap_threshold : float
        The gap fraction above which a column is flagged.
    """

    n_rows: int
    n_cols: int
    gap_fraction: np.ndarray
    ambiguous_fraction: np.ndarray
    context_unreliable: np.ndarray
    gap_threshold: float

    @property
    def n_flagged(self) -> int:
        """
        Number of columns flagged as context-unreliable.

        Returns
        -------
        int
            Count of flagged columns.
        """
        return int(self.context_unreliable.sum())


def profile_alignment(alignment, gap_threshold: float = 0.5) -> ColumnProfile:
    """
    Compute the per-column gap and ambiguity profile of an alignment.

    Parameters
    ----------
    alignment : Bio.Align.MultipleSeqAlignment
        The alignment to profile.
    gap_threshold : float, optional
        Fraction of gaps above which a column is flagged context-unreliable
        (default: 0.5).

    Returns
    -------
    ColumnProfile
        The computed profile.
    """
    from derip2.aln_ops import alignment_to_array

    arr = alignment_to_array(alignment)
    n_rows, n_cols = arr.shape
    codes = arr.view(np.uint8)

    is_gap = arr == b'-'
    is_acgt = np.isin(codes, np.frombuffer(bytes(_ACGT), dtype=np.uint8))
    is_ambiguous = ~is_gap & ~is_acgt

    gap_fraction = is_gap.sum(axis=0) / n_rows
    ambiguous_fraction = is_ambiguous.sum(axis=0) / n_rows
    context_unreliable = gap_fraction > gap_threshold

    logger.info(
        'QC: %d columns, %d flagged as context-unreliable (>%.0f%% gaps)',
        n_cols,
        int(context_unreliable.sum()),
        gap_threshold * 100,
    )
    return ColumnProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        gap_fraction=gap_fraction,
        ambiguous_fraction=ambiguous_fraction,
        context_unreliable=context_unreliable,
        gap_threshold=gap_threshold,
    )


def write_column_profile(profile: ColumnProfile, path: str) -> str:
    """
    Write the per-column gap/ambiguity profile to a TSV file.

    Parameters
    ----------
    profile : ColumnProfile
        The profile to write.
    path : str
        Destination path.

    Returns
    -------
    str
        The path written.
    """
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('column\tgap_fraction\tambiguous_fraction\tcontext_unreliable\n')
        for col in range(profile.n_cols):
            handle.write(
                f'{col}\t{profile.gap_fraction[col]:.4f}\t'
                f'{profile.ambiguous_fraction[col]:.4f}\t'
                f'{int(profile.context_unreliable[col])}\n'
            )
    return path


def write_qc_report(alignment, profile: ColumnProfile, path: str) -> str:
    """
    Write a short human-readable QC summary.

    Parameters
    ----------
    alignment : Bio.Align.MultipleSeqAlignment
        The alignment that was profiled.
    profile : ColumnProfile
        The computed profile.
    path : str
        Destination path.

    Returns
    -------
    str
        The path written.
    """
    lines: List[str] = [
        'deRIP2 mutation-spectrum QC report',
        '=' * 36,
        f'Sequences:            {profile.n_rows}',
        f'Alignment columns:    {profile.n_cols}',
        f'Gap threshold:        {profile.gap_threshold:.2f}',
        f'Context-unreliable:   {profile.n_flagged} columns',
        f'Mean gap fraction:    {profile.gap_fraction.mean():.4f}',
        f'Max gap fraction:     {profile.gap_fraction.max():.4f}',
        f'Mean ambiguity:       {profile.ambiguous_fraction.mean():.4f}',
    ]
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    return path
