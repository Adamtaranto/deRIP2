"""
Branch-by-branch substitution calling for the phylogenetic mutation spectrum.

Given a rooted, ancestrally-reconstructed tree (:class:`TreeReconstruction`), this
walks every directed parent -> child edge and logs each column where the two
sequences differ as one independent substitution event. The trinucleotide context
is read from the **parent** sequence at that branch — the sequence state at the
moment the mutation occurred — using the nearest non-gap bases.

Counting events per edge, rather than per tip against one reference, is what makes
recurrent (homoplasic) deamination visible: the same C>T arising independently on
three branches is three events here, and the homoplasy table records that a column
was hit on multiple independent branches.

The assembled event stream is fed to the same channel-assembly core
(:func:`derip2.stats.mutation_spectra.assemble_matrices`) as the tree-free
baseline, so the two methods produce directly comparable SBS-96 / SBS-192
matrices.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from derip2.spectra.channels import downstream_context, trinucleotide_context
from derip2.stats.mutation_spectra import (
    SpectraResult,
    assemble_downstream,
    assemble_matrices,
)

logger = logging.getLogger(__name__)

# Base-code lookups, matching derip2.stats.mutation_spectra.
_CODE_LUT = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate((b'A', b'C', b'G', b'T')):
    _CODE_LUT[_b[0]] = _i
    _CODE_LUT[_b.lower()[0]] = _i
_CODE_TO_BASE = np.array([b'A', b'C', b'G', b'T'], dtype='S1')


def _resolve_branch_samples(samples_by_child, edges):
    """
    Map each edge's child to a sample column.

    Parameters
    ----------
    samples_by_child : dict of str to str or None
        Optional mapping from child node name to sample label. When ``None`` every
        edge is pooled into a single ``AllBranches`` sample.
    edges : list of tuple of str
        Directed ``(parent, child)`` edges.

    Returns
    -------
    tuple
        ``(sample_names, child_to_sample)`` where ``child_to_sample`` maps a child
        name to its integer sample index.
    """
    if samples_by_child is None:
        names = ['AllBranches']
        return names, {child: 0 for _parent, child in edges}
    names: List[str] = []
    index: Dict[str, int] = {}
    child_to_sample: Dict[str, int] = {}
    for _parent, child in edges:
        label = samples_by_child.get(child, 'trunk')
        if label not in index:
            index[label] = len(names)
            names.append(label)
        child_to_sample[child] = index[label]
    return names, child_to_sample


class _ParentContext:
    """
    Cached per-parent sanitised sequence, base codes and sequence context.

    Parameters
    ----------
    seq : numpy.ndarray
        The parent node's ``(n_cols,)`` ``'S1'`` sequence.
    context : {'trinucleotide', 'downstream'}, optional
        Which context to resolve per column (default: ``'trinucleotide'``). For
        ``'trinucleotide'`` the two codes are the 5'/3' flanks; for ``'downstream'``
        they are the two pyrimidine-strand downstream bases.

    Attributes
    ----------
    code : numpy.ndarray
        ``(n_cols,)`` base codes (``-1`` for gap/ambiguous).
    ctx1_code, ctx2_code : numpy.ndarray
        ``(n_cols,)`` context base codes from the nearest non-gap bases (``-1``
        where unresolved): the 5'/3' flanks (trinucleotide) or the two downstream
        bases (downstream).
    context_ok : numpy.ndarray
        ``(n_cols,)`` boolean; True where a full context resolved.
    """

    __slots__ = ('code', 'ctx1_code', 'ctx2_code', 'context_ok')

    def __init__(self, seq: np.ndarray, context: str = 'trinucleotide'):
        from derip2.aln_ops import _nongap_neighbors

        resolve = (
            downstream_context if context == 'downstream' else trinucleotide_context
        )
        n_cols = seq.shape[0]
        code = _CODE_LUT[seq.view(np.uint8)]
        sanitized = np.where(code >= 0, seq, b'-')
        next_idx, prev_idx = _nongap_neighbors(sanitized.reshape(1, n_cols))
        next_idx = next_idx[0]
        prev_idx = prev_idx[0]

        self.code = code.astype(np.int64)
        self.ctx1_code = np.full(n_cols, -1, dtype=np.int64)
        self.ctx2_code = np.full(n_cols, -1, dtype=np.int64)
        self.context_ok = np.zeros(n_cols, dtype=bool)
        for col in range(n_cols):
            if code[col] < 0:
                continue
            ctx = resolve(sanitized, col, next_idx, prev_idx)
            if ctx is None:
                continue
            ctx1, ctx2 = ctx
            self.ctx1_code[col] = _CODE_LUT[ord(ctx1)]
            self.ctx2_code[col] = _CODE_LUT[ord(ctx2)]
            self.context_ok[col] = True


def compute_spectra_from_tree(
    reconstruction,
    *,
    samples_by_child: Optional[Dict[str, str]] = None,
    min_prob: float = 0.0,
    context: str = 'trinucleotide',
) -> SpectraResult:
    """
    Call substitutions along every branch and assemble the mutation spectra.

    Parameters
    ----------
    reconstruction : derip2.spectra.tree_asr.TreeReconstruction
        A rooted, oriented, ancestrally-reconstructed tree.
    samples_by_child : dict of str to str, optional
        Maps a child node name to a sample label, so events can be partitioned by
        clade. Children absent from the map fall into a ``trunk`` sample. Default
        ``None`` pools all branches into one ``AllBranches`` sample.
    min_prob : float, optional
        Drop events whose combined parent/child state posterior probability is
        below this threshold (default ``0.0``, keep all).
    context : {'trinucleotide', 'downstream'}, optional
        Which sequence context to classify substitutions by (default:
        ``'trinucleotide'``). ``'downstream'`` builds the pyrimidine-folded
        downstream-triplet matrix and leaves ``sbs192`` ``None``.

    Returns
    -------
    derip2.stats.mutation_spectra.SpectraResult
        The phylogenetic spectra, per-event detail (with parent/child names) and
        the true (per-branch) homoplasy counts.
    """
    downstream = context == 'downstream'
    edges = reconstruction.edges
    node_seq = reconstruction.node_seq
    node_prob = reconstruction.node_prob
    n_cols = reconstruction.n_cols

    sample_names, child_to_sample = _resolve_branch_samples(samples_by_child, edges)
    node_index = {name: i for i, name in enumerate(node_seq)}

    parent_cache: Dict[str, _ParentContext] = {}

    cols_all: List[np.ndarray] = []
    ref_all: List[np.ndarray] = []
    alt_all: List[np.ndarray] = []
    ctx1_all: List[np.ndarray] = []
    ctx2_all: List[np.ndarray] = []
    sample_all: List[np.ndarray] = []
    weight_all: List[np.ndarray] = []
    child_ord_all: List[np.ndarray] = []
    parent_names: List[str] = []
    child_names: List[str] = []

    n_indel_or_ambiguous = 0
    n_unassignable_context = 0

    for parent, child in edges:
        if parent not in parent_cache:
            parent_cache[parent] = _ParentContext(node_seq[parent], context=context)
        pc = parent_cache[parent]

        child_seq = node_seq[child]
        child_code = _CODE_LUT[child_seq.view(np.uint8)].astype(np.int64)

        both_valid = (pc.code >= 0) & (child_code >= 0)
        differ = pc.code != child_code
        substitution = both_valid & differ
        n_indel_or_ambiguous += int((differ & ~both_valid).sum())

        countable = substitution & pc.context_ok
        n_unassignable_context += int((substitution & ~pc.context_ok).sum())

        cols = np.nonzero(countable)[0]
        if cols.size == 0:
            continue

        weight = node_prob[parent][cols] * node_prob[child][cols]
        keep = weight >= min_prob
        if not keep.all():
            cols = cols[keep]
            weight = weight[keep]
        if cols.size == 0:
            continue

        cols_all.append(cols)
        ref_all.append(pc.code[cols])
        alt_all.append(child_code[cols])
        ctx1_all.append(pc.ctx1_code[cols])
        ctx2_all.append(pc.ctx2_code[cols])
        sample_all.append(np.full(cols.size, child_to_sample[child], dtype=np.int64))
        weight_all.append(weight)
        child_ord_all.append(np.full(cols.size, node_index[child], dtype=np.int64))
        parent_names.extend([parent] * cols.size)
        child_names.extend([child] * cols.size)

    if cols_all:
        cols = np.concatenate(cols_all)
        ref_c = np.concatenate(ref_all)
        alt_c = np.concatenate(alt_all)
        ctx1_c = np.concatenate(ctx1_all)
        ctx2_c = np.concatenate(ctx2_all)
        sample_c = np.concatenate(sample_all)
        child_ord = np.concatenate(child_ord_all)
    else:
        cols = ref_c = alt_c = ctx1_c = ctx2_c = sample_c = child_ord = np.array(
            [], dtype=np.int64
        )

    if downstream:
        # Single pyrimidine-folded downstream matrix; no strand-resolved form.
        sbs96 = assemble_downstream(
            ctx1_c, ref_c, alt_c, ctx2_c, sample_c, len(sample_names)
        )
        sbs192 = None
    else:
        sbs96, sbs192 = assemble_matrices(
            ctx1_c, ref_c, alt_c, ctx2_c, sample_c, len(sample_names)
        )

    # True homoplasy: independent branch events per column per derived base.
    homoplasy_counts = np.zeros((n_cols, 4), dtype=np.int64)
    if cols.size:
        np.add.at(homoplasy_counts, (cols, alt_c), 1)

    root_seq = node_seq[reconstruction.root_name]
    root_code = _CODE_LUT[root_seq.view(np.uint8)]
    ancestor_ref = np.where(root_code >= 0, root_seq, b'-')

    logger.info(
        'Branch traversal: %d events on %d edges across %d sample(s) '
        '(%d indel/ambiguous, %d unassignable context)',
        cols.size,
        len(edges),
        len(sample_names),
        n_indel_or_ambiguous,
        n_unassignable_context,
    )

    empty = np.array([], dtype='S1')
    ctx1_bytes = _CODE_TO_BASE[ctx1_c] if cols.size else empty
    ctx2_bytes = _CODE_TO_BASE[ctx2_c] if cols.size else empty
    return SpectraResult(
        sbs96=sbs96,
        sbs192=sbs192,
        sample_names=sample_names,
        event_rows=child_ord,
        event_cols=cols,
        event_ref=_CODE_TO_BASE[ref_c] if cols.size else empty,
        event_alt=_CODE_TO_BASE[alt_c] if cols.size else empty,
        event_five=None if downstream else ctx1_bytes,
        event_three=None if downstream else ctx2_bytes,
        event_down1=ctx1_bytes if downstream else None,
        event_down2=ctx2_bytes if downstream else None,
        context=context,
        event_sample=sample_c,
        homoplasy_counts=homoplasy_counts,
        ancestor_ref=ancestor_ref,
        n_indel_or_ambiguous=n_indel_or_ambiguous,
        n_unassignable_context=n_unassignable_context,
        method='phylogenetic',
        event_parent_names=parent_names,
        event_child_names=child_names,
    )
