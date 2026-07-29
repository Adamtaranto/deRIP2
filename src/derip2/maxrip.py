"""
Maximum-RIP counterfactuals of the deRIP'd consensus.

Where :mod:`derip2.aln_ops` runs RIP *backwards* — restoring the ancestral C and G
that RIP deaminated — this module runs it forwards to exhaustion. Given the deRIP'd
consensus it produces the sequence that would result if every RIP target it still
carries had been mutated, answering "how far could this locus have gone?" rather
than "where did it start?".

Three variants are offered, differing only in which sites count as targets:

``'all'``
    Every RIP substrate site present in the consensus: a ``C`` whose 3' neighbour
    is ``A`` becomes ``T``, and a ``G`` whose 5' neighbour is ``T`` becomes ``A``.
    This is the pure counterfactual and ignores the alignment entirely.
``'observed'``
    The same rule, but restricted to alignment columns where RIP demonstrably
    occurred in at least one input sequence. Conservative: it will not invent
    mutation at a site the family gives no evidence for.
``'all_plus_nonrip'``
    ``'all'``, plus the columns where the alignment shows deamination *outside*
    RIP dinucleotide context. Models a locus subject to both RIP and
    context-independent cytosine deamination.

**Context comes from the consensus; evidence comes from the alignment.** The
deRIP'd consensus is a chimera — conserved columns, then RIP corrections, then a
fill from a chosen reference row — so a restored ``C`` can end up beside a filled
``A`` and form a ``CpA`` that exists in no single input sequence. Since these
variants are claims about the *reconstructed ancestor*, the dinucleotide context is
read off the consensus itself; the per-column masks of
:class:`derip2.aln_ops.ColumnClassification` are consulted only to decide which
columns carry evidence (the ``'observed'`` and ``'all_plus_nonrip'`` filters).

Gap handling matches the alignment scan exactly, by reusing
:func:`derip2.aln_ops._nongap_neighbors`: a ``C-A`` spanning a gap column is a
substrate site, just as it is when the alignment is classified.
"""

from dataclasses import dataclass
import logging
from typing import Union

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

from derip2.aln_ops import ColumnClassification, _nongap_neighbors

logger = logging.getLogger(__name__)

#: The supported maximum-RIP variants, in increasing order of aggressiveness.
MAX_RIP_VARIANTS = ('observed', 'all', 'all_plus_nonrip')

#: Human-readable one-line description of each variant, for reports and CLI help.
MAX_RIP_DESCRIPTIONS = {
    'all': (
        'Every RIP substrate site in the deRIP sequence converted to product '
        '(CpA to TpA, TpG to TpA).'
    ),
    'observed': (
        'Only substrate sites in alignment columns where RIP was observed in at '
        'least one input sequence.'
    ),
    'all_plus_nonrip': (
        'Every RIP substrate site, plus deamination at columns where the '
        'alignment shows C to T change outside RIP dinucleotide context.'
    ),
}


@dataclass(frozen=True)
class MaxRIPResult:
    """
    A maximally RIP-mutated variant of the deRIP'd consensus.

    Attributes
    ----------
    variant : str
        Which rule produced this sequence; one of :data:`MAX_RIP_VARIANTS`.
    gapped_seq : str
        The mutated consensus with gap columns preserved, so it stays aligned
        column-for-column with the input alignment.
    seq : str
        ``gapped_seq`` with gaps removed.
    converted_cols : numpy.ndarray
        Ascending alignment column indices that were mutated (``int64``).
    converted_positions : numpy.ndarray
        The same sites as zero-based offsets into ``seq`` (``int64``).
    strand : numpy.ndarray
        ``'S1'`` array of ``b'+'`` / ``b'-'``, parallel to ``converted_cols``,
        recording whether each site was a forward (``CpA``) or reverse (``TpG``)
        target.
    n_forward : int
        Number of forward-strand conversions.
    n_reverse : int
        Number of reverse-strand conversions.
    """

    variant: str
    gapped_seq: str
    seq: str
    converted_cols: np.ndarray
    converted_positions: np.ndarray
    strand: np.ndarray
    n_forward: int
    n_reverse: int

    @property
    def n_converted(self) -> int:
        """
        Total number of converted sites.

        Returns
        -------
        int
            ``n_forward + n_reverse``.
        """
        return self.n_forward + self.n_reverse

    def as_record(self, seq_id: str = 'maxRIPseq', gapped: bool = False) -> SeqRecord:
        """
        Wrap the sequence as a :class:`~Bio.SeqRecord.SeqRecord` for writing.

        Parameters
        ----------
        seq_id : str, optional
            Record id and name (default ``'maxRIPseq'``).
        gapped : bool, optional
            Emit the column-aligned sequence rather than the ungapped one
            (default ``False``).

        Returns
        -------
        Bio.SeqRecord.SeqRecord
            The sequence, with the variant named in its description.
        """
        return SeqRecord(
            Seq(self.gapped_seq if gapped else self.seq),
            id=seq_id,
            name=seq_id,
            description=f'maximum RIP ({self.variant}): {self.n_converted} sites',
        )


def _consensus_array(gapped_consensus: Union[str, SeqRecord, Seq]) -> np.ndarray:
    """
    Coerce a gapped consensus to an upper-case ``(1, n_cols)`` byte array.

    Parameters
    ----------
    gapped_consensus : str or Bio.Seq.Seq or Bio.SeqRecord.SeqRecord
        The column-aligned deRIP'd consensus.

    Returns
    -------
    numpy.ndarray
        ``(1, n_cols)`` ``'S1'`` array, shaped like a one-row alignment so the
        alignment helpers can be reused unchanged.
    """
    if isinstance(gapped_consensus, SeqRecord):
        text = str(gapped_consensus.seq)
    else:
        text = str(gapped_consensus)
    return np.frombuffer(text.upper().encode('ascii'), dtype='S1')[None, :]


def compute_max_rip(
    gapped_consensus: Union[str, SeqRecord, Seq],
    cls: ColumnClassification,
    *,
    variant: str = 'all',
) -> MaxRIPResult:
    """
    Build a maximally RIP-mutated variant of a deRIP'd consensus.

    Substrate context is read from ``gapped_consensus`` itself; ``cls`` supplies
    only the per-column evidence used by the ``'observed'`` and
    ``'all_plus_nonrip'`` variants. See the module docstring for why.

    Conversion runs to a fixed point, so the result is stable: re-running this
    function on its own output converts nothing.

    For ``'all'`` and ``'observed'`` a single pass already is that fixed point,
    because a context-derived conversion cannot create a new target. A forward
    ``C`` would need its 3' neighbour to *become* ``A``, which only a reverse
    ``G``-to-``A`` does, and that requires the base before the ``G`` to be ``T``
    — but here it is the ``C`` itself; the reverse case is symmetric.

    ``'all_plus_nonrip'`` genuinely does cascade, because its extra sites are
    context-free: a ``C`` converted to ``T`` immediately 5' of a ``G`` creates a
    ``TpG`` that was not a substrate beforehand. A maximally mutated sequence
    should carry that downstream conversion too, so the passes repeat until
    nothing changes. Each pass strictly reduces the number of ``C`` and ``G``, so
    the loop always terminates.

    Parameters
    ----------
    gapped_consensus : str or Bio.Seq.Seq or Bio.SeqRecord.SeqRecord
        The column-aligned deRIP'd consensus, the same length as the alignment.
    cls : derip2.aln_ops.ColumnClassification
        The alignment's cached column classification.
    variant : {'all', 'observed', 'all_plus_nonrip'}, optional
        Which rule to apply (default ``'all'``).

    Returns
    -------
    MaxRIPResult
        The mutated sequence and the sites that were changed.

    Raises
    ------
    ValueError
        If ``variant`` is unknown, or if the consensus length does not match the
        alignment width recorded in ``cls``.
    """
    if variant not in MAX_RIP_VARIANTS:
        raise ValueError(
            f'Unknown maximum-RIP variant {variant!r}; '
            f'expected one of {", ".join(MAX_RIP_VARIANTS)}'
        )

    arr2d = _consensus_array(gapped_consensus)
    n_cols = cls.arr.shape[1]
    if arr2d.shape[1] != n_cols:
        raise ValueError(
            f'Consensus length {arr2d.shape[1]} does not match the alignment '
            f'width {n_cols}'
        )
    bases = arr2d[0]

    # Gap-aware neighbours, computed exactly as the alignment scan computes them,
    # so a substrate site split by a gap column is still recognised. Gap columns
    # never change, so these indices are computed once and stay valid.
    next_idx, prev_idx = _nongap_neighbors(arr2d)
    next_idx, prev_idx = next_idx[0], prev_idx[0]
    # Clip the -1 sentinels before indexing, then mask those cells back out: a
    # base with no neighbour on the relevant side can never be a substrate site.
    has_next = next_idx >= 0
    has_prev = prev_idx >= 0
    safe_next = np.where(has_next, next_idx, 0)
    safe_prev = np.where(has_prev, prev_idx, 0)

    # Columns whose evidence lets a variant convert regardless of context. Only
    # 'all_plus_nonrip' has any: a non-RIP deamination column is by definition
    # one where C->T happened *outside* CpA context, so requiring CpA there would
    # make the clause unreachable.
    if variant == 'all_plus_nonrip':
        free_fwd = cls.nonrip_fwd.any(axis=0)
        free_rev = cls.nonrip_rev.any(axis=0)
    else:
        free_fwd = np.zeros(n_cols, dtype=bool)
        free_rev = np.zeros(n_cols, dtype=bool)

    def targets(current):
        """
        Find the sites this variant would convert in ``current``.

        Parameters
        ----------
        current : numpy.ndarray
            ``(n_cols,)`` ``'S1'`` array of the sequence as it now stands.

        Returns
        -------
        tuple of numpy.ndarray
            ``(fwd, rev)`` boolean masks of forward and reverse targets.
        """
        is_c = current == b'C'
        is_g = current == b'G'
        sub_fwd = is_c & has_next & (current[safe_next] == b'A')  # CpA -> TpA
        sub_rev = is_g & has_prev & (current[safe_prev] == b'T')  # TpG -> TpA
        if variant == 'observed':
            # Only where the alignment itself shows RIP at this column.
            return sub_fwd & cls.fwd_col, sub_rev & cls.rev_col
        return sub_fwd | (is_c & free_fwd), sub_rev | (is_g & free_rev)

    # Run to a fixed point. For 'all' and 'observed' a single pass is provably
    # enough (a conversion cannot create a new target -- see the docstring), so
    # the second pass just confirms it. 'all_plus_nonrip' genuinely can cascade:
    # a context-free C->T immediately 5' of a G creates a TpG that was not a
    # substrate before, and a maximally RIP'd sequence should carry that too.
    # Each pass strictly reduces the number of C and G, so this terminates.
    out = bases.copy()
    fwd_hits = np.zeros(n_cols, dtype=bool)
    rev_hits = np.zeros(n_cols, dtype=bool)
    for _ in range(n_cols + 1):
        fwd, rev = targets(out)
        # A base is either C or G, never both, so the target sets cannot overlap.
        assert not (fwd & rev).any(), 'forward and reverse targets overlap'
        if not (fwd.any() or rev.any()):
            break
        out[fwd] = b'T'
        out[rev] = b'A'
        fwd_hits |= fwd
        rev_hits |= rev
    else:  # pragma: no cover - unreachable; each pass removes at least one C/G
        raise RuntimeError('maximum-RIP conversion failed to reach a fixed point')

    gapped_seq = out.tobytes().decode('ascii')

    converted = fwd_hits | rev_hits
    converted_cols = np.flatnonzero(converted).astype(np.int64)
    # Offset of each column within the ungapped sequence. A converted column is
    # never a gap (only C and G are targets), so the lookup is always valid.
    ungapped_of = np.cumsum(bases != b'-') - 1
    converted_positions = ungapped_of[converted_cols].astype(np.int64)
    strand = np.where(fwd_hits[converted_cols], b'+', b'-').astype('S1')

    n_forward = int(fwd_hits.sum())
    n_reverse = int(rev_hits.sum())
    logger.debug(
        f'Maximum-RIP variant {variant!r}: converted {n_forward} forward and '
        f'{n_reverse} reverse sites across {n_cols} columns'
    )

    return MaxRIPResult(
        variant=variant,
        gapped_seq=gapped_seq,
        seq=gapped_seq.replace('-', ''),
        converted_cols=converted_cols,
        converted_positions=converted_positions,
        strand=strand,
        n_forward=n_forward,
        n_reverse=n_reverse,
    )


def write_max_rip_fasta(
    results,
    output_file: str,
    seq_id: str = 'maxRIPseq',
    gapped: bool = False,
) -> str:
    """
    Write one or more maximum-RIP sequences to a multi-FASTA file.

    Parameters
    ----------
    results : iterable of MaxRIPResult
        The sequences to write, in the order they should appear.
    output_file : str
        Destination path.
    seq_id : str, optional
        Base record id; each record is suffixed with its variant name so the
        records stay uniquely identifiable (default ``'maxRIPseq'``).
    gapped : bool, optional
        Write the column-aligned sequences rather than the ungapped ones
        (default ``False``).

    Returns
    -------
    str
        ``output_file``, for chaining.
    """
    from Bio import SeqIO

    records = [
        result.as_record(seq_id=f'{seq_id}_{result.variant}', gapped=gapped)
        for result in results
    ]
    with open(output_file, 'w', encoding='utf-8') as handle:
        SeqIO.write(records, handle, 'fasta')
    logger.info(f'Wrote {len(records)} maximum-RIP sequence(s) to {output_file}')
    return output_file


def max_rip_multifasta(results, seq_id: str = 'maxRIPseq', width: int = 60) -> str:
    """
    Render maximum-RIP sequences as a plain multi-FASTA string.

    Used by the HTML report's download link, which embeds the text directly
    rather than writing a file.

    Parameters
    ----------
    results : iterable of MaxRIPResult
        The sequences to render.
    seq_id : str, optional
        Base record id; suffixed with each variant name (default ``'maxRIPseq'``).
    width : int, optional
        Line-wrap width (default: 60).

    Returns
    -------
    str
        A FASTA document; empty when ``results`` is empty.
    """
    out = []
    for result in results:
        seq = result.seq
        lines = [seq[i : i + width] for i in range(0, len(seq), width)] or ['']
        body = '\n'.join(lines)
        out.append(
            f'>{seq_id}_{result.variant} {MAX_RIP_DESCRIPTIONS[result.variant]}\n'
            f'{body}\n'
        )
    return ''.join(out)
