"""
RIP Strandedness Imbalance (RSI): a per-sequence measure of which strand RIP acted on.

Notes
-----
**Background**

RIP deaminates the C of a CpA dinucleotide. Read on the forward strand, a
reverse-strand CpA appears as TpG, so RIP acting on either strand converts a
forward-strand dinucleotide to TpA::

    forward substrate   CA  --RIP-->  TA
    reverse substrate   TG  --RIP-->  TA

A single round of meiotic RIP acts on one strand of a given duplex, so progeny
sequences carry a strand-biased signature: either their CpA sites converted, or
their TpG sites converted, rarely both. RSI quantifies that asymmetry.

**Definition**

For each sequence::

    p_fwd = fwd_products / (fwd_products + fwd_substrates)
    p_rev = rev_products / (rev_products + rev_substrates)
    RSI   = p_fwd - p_rev

RSI lies in ``[-1, 1]``. Positive values indicate RIP predominantly on the
forward strand, negative values the reverse strand. Both ``0`` (no RIP) and
``0`` (both strands fully converted) are neutral, so RSI must be read alongside
its components ``p_fwd`` and ``p_rev``, which distinguish the two cases.

Because the proportions are normalised independently per strand, unequal
abundances of CpA and TpG substrate motifs do not bias the score.

**Substrates versus products**

Unmutated substrates (CA, TG) are *directly observed* in a sequence. Products
(TA) must be *inferred*: a TA is only attributable to RIP when the alignment
column shows an aligned, unmutated substrate in some other sequence. The two
are therefore counted with different scopes, controlled by ``substrate_scope``.

Counting substrates only inside RIP-classified columns would give a sequence
with no RIP at all ``p_fwd = 0 / 0``, when the correct answer is ``0``.

**Ambiguity**

A physical TA dinucleotide spans two columns: the T at column ``i`` and the A
at column ``j``. It is evidence of forward RIP if column ``i`` is a forward RIP
column (some sequence retains an aligned CA), and evidence of reverse RIP if
column ``j`` is a reverse RIP column (some sequence retains an aligned TG). When
both hold the strand of origin is unrecoverable from the alignment alone. The
``ambiguous`` policy decides how such events are attributed; ``n_ambiguous`` is
always reported so the choice can be audited.

Substrates are never ambiguous: a CA's second base is A (not G) and a TG's first
base is T (not C), so neither can be read as the other strand's substrate.
"""

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from derip2.aln_ops import ColumnClassification

AMBIGUITY_POLICIES = ('split', 'exclude', 'weight', 'both')
SUBSTRATE_SCOPES = ('all', 'assessable', 'rip_like_columns')


@dataclass(frozen=True)
class RSIResult:
    """
    Per-sequence RIP strandedness imbalance and its components.

    All arrays have shape ``(n_rows,)`` and are indexed by alignment row.

    Attributes
    ----------
    rsi : numpy.ndarray
        ``p_fwd - p_rev``, in ``[-1, 1]``. NaN when either strand has no
        evidence.
    p_fwd, p_rev : numpy.ndarray
        Proportion of available forward (reverse) substrate sites converted to
        product. NaN when that strand has neither substrate nor product.
    fwd_prod, rev_prod : numpy.ndarray
        Attributed product counts. Fractional under the ``'split'`` and
        ``'weight'`` policies.
    fwd_sub, rev_sub : numpy.ndarray
        Counts of unmutated substrate dinucleotides.
    n_ambiguous : numpy.ndarray
        Number of TA dinucleotides attributable to either strand.
    z : numpy.ndarray
        Two-proportion z statistic for ``p_fwd`` vs ``p_rev``.
    pvalue : numpy.ndarray
        Two-sided p-value for the null that RIP struck both strands equally.
    ambiguous : str
        The attribution policy used.
    substrate_scope : str
        The substrate counting scope used.
    """

    rsi: np.ndarray
    p_fwd: np.ndarray
    p_rev: np.ndarray
    fwd_prod: np.ndarray
    fwd_sub: np.ndarray
    rev_prod: np.ndarray
    rev_sub: np.ndarray
    n_ambiguous: np.ndarray
    z: np.ndarray
    pvalue: np.ndarray
    ambiguous: str
    substrate_scope: str

    def pooled(self):
        """
        Pool counts across all sequences and recompute the imbalance.

        Summing the raw counts before taking the ratios weights each sequence by
        how many informative sites it carries, unlike the mean of the per-row
        RSI values, which weights every sequence equally regardless of evidence.

        Returns
        -------
        dict
            ``p_fwd``, ``p_rev``, ``RSI``, the four pooled counts,
            ``n_ambiguous``, ``z`` and ``pvalue`` for the alignment as a whole.
            Proportions are NaN when the corresponding denominator is zero.
        """
        fp, fs = float(self.fwd_prod.sum()), float(self.fwd_sub.sum())
        rp, rs = float(self.rev_prod.sum()), float(self.rev_sub.sum())
        den_f, den_r = fp + fs, rp + rs

        p_fwd = fp / den_f if den_f > 0 else np.nan
        p_rev = rp / den_r if den_r > 0 else np.nan

        z, pvalue = _two_proportion_test(
            np.array([fp]), np.array([den_f]), np.array([rp]), np.array([den_r])
        )
        return {
            'p_fwd': p_fwd,
            'p_rev': p_rev,
            'RSI': p_fwd - p_rev,
            'fwd_product': fp,
            'fwd_substrate': fs,
            'rev_product': rp,
            'rev_substrate': rs,
            'n_ambiguous': int(self.n_ambiguous.sum()),
            'z': float(z[0]),
            'pvalue': float(pvalue[0]),
        }

    def as_records(self, ids=None):
        """
        Return the result as a list of per-sequence dictionaries.

        Parameters
        ----------
        ids : sequence of str, optional
            Sequence identifiers, one per row. Defaults to row indices.

        Returns
        -------
        list of dict
            One dictionary per sequence, in alignment order.
        """
        n = self.rsi.size
        ids = list(ids) if ids is not None else [str(i) for i in range(n)]
        return [
            {
                'index': i,
                'id': ids[i],
                'RSI': float(self.rsi[i]),
                'p_fwd': float(self.p_fwd[i]),
                'p_rev': float(self.p_rev[i]),
                'fwd_product': float(self.fwd_prod[i]),
                'fwd_substrate': float(self.fwd_sub[i]),
                'rev_product': float(self.rev_prod[i]),
                'rev_substrate': float(self.rev_sub[i]),
                'n_ambiguous': int(self.n_ambiguous[i]),
                'z': float(self.z[i]),
                'pvalue': float(self.pvalue[i]),
            }
            for i in range(n)
        ]


def _two_proportion_test(x1, n1, x2, n2):
    """
    Two-sided two-proportion z-test, vectorised over rows.

    Tests the null hypothesis that the forward and reverse conversion
    proportions are equal.

    Parameters
    ----------
    x1, n1 : numpy.ndarray
        Forward successes and trials, shape ``(n_rows,)``.
    x2, n2 : numpy.ndarray
        Reverse successes and trials, shape ``(n_rows,)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(z, pvalue)``. Both NaN where either denominator is zero. Where the
        pooled proportion is 0 or 1 the two proportions are necessarily equal,
        giving ``z = 0`` and ``pvalue = 1``.

    Notes
    -----
    The test assumes integer counts. Under the ``'split'`` and ``'weight'``
    ambiguity policies the inputs are fractional, so the p-value is an
    approximation. The ``'both'`` policy keeps counts integral but double-counts
    ambiguous events, which inflates ``n1`` and ``n2`` and so overstates
    significance. Treat these p-values as a screening heuristic, not as a
    calibrated test.
    """
    n_rows = x1.size
    z = np.full(n_rows, np.nan)
    pvalue = np.full(n_rows, np.nan)

    valid = (n1 > 0) & (n2 > 0)
    if not valid.any():
        return z, pvalue

    p1 = np.divide(x1, n1, out=np.zeros(n_rows), where=valid)
    p2 = np.divide(x2, n2, out=np.zeros(n_rows), where=valid)
    p_pool = np.divide(x1 + x2, n1 + n2, out=np.zeros(n_rows), where=valid)

    se = np.sqrt(
        p_pool
        * (1.0 - p_pool)
        * (1.0 / np.where(valid, n1, 1) + 1.0 / np.where(valid, n2, 1))
    )

    # SE == 0 means every trial succeeded or every trial failed on both strands,
    # so p1 == p2 exactly and there is no evidence of asymmetry.
    degenerate = valid & (se == 0)
    testable = valid & (se > 0)

    z[degenerate] = 0.0
    pvalue[degenerate] = 1.0

    z[testable] = (p1[testable] - p2[testable]) / se[testable]
    pvalue[testable] = [math.erfc(abs(v) / math.sqrt(2.0)) for v in z[testable]]

    return z, pvalue


def compute_rsi(
    cls: 'ColumnClassification',
    ambiguous: str = 'split',
    substrate_scope: str = 'all',
) -> RSIResult:
    """
    Compute the RIP Strandedness Imbalance for every sequence in an alignment.

    Parameters
    ----------
    cls : ColumnClassification
        Classification produced by :func:`derip2.aln_ops.classify_columns`.
    ambiguous : {'split', 'exclude', 'weight', 'both'}, optional
        How to attribute TA dinucleotides that could have arisen from RIP on
        either strand (default: ``'split'``).

        - ``'split'``: contribute 0.5 to each strand. Uses all the data and is
          unbiased when ambiguity is strand-symmetric.
        - ``'exclude'``: drop from both strands. Most conservative; discards the
          products of heavily RIP'd sequences.
        - ``'weight'``: split in proportion to the evidence, giving the forward
          strand ``nC(i) / (nC(i) + nG(j))`` where ``nC(i)`` is the count of
          unmutated C at the T's column and ``nG(j)`` the count of unmutated G
          at the A's column.
        - ``'both'``: contribute 1.0 to each strand. Keeps counts integral but
          inflates both proportions toward 1.
    substrate_scope : {'all', 'assessable', 'rip_like_columns'}, optional
        Which unmutated substrate dinucleotides enter the denominators
        (default: ``'all'``).

        - ``'all'``: every observed CA and TG in the sequence. Substrates are
          directly observed and need no column-level inference.
        - ``'assessable'``: only those in columns passing the ``max_snp_noise``
          gate. Matches the scope of ``markupdict['rip_substrate']``.
        - ``'rip_like_columns'``: only those in columns that also contain a
          product. Symmetric with the product scope, but yields NaN for
          sequences with no RIP.

    Returns
    -------
    RSIResult
        Per-sequence RSI, components, ambiguity counts and significance.

    Raises
    ------
    ValueError
        If ``ambiguous`` or ``substrate_scope`` is not a recognised option.

    Notes
    -----
    Products are always counted only within RIP columns (``fwd_col`` /
    ``rev_col``), because a TA can only be attributed to RIP when an aligned
    sequence retains the unmutated substrate.

    A denominator of zero yields NaN rather than 0: a strand with neither
    substrate nor product carries no evidence, and reporting 0 would disguise
    "no data" as "perfectly one-sided".

    Examples
    --------
    >>> from derip2.aln_ops import classify_alignment
    >>> from derip2.stats import compute_rsi
    >>> cls = classify_alignment(alignment)          # doctest: +SKIP
    >>> res = compute_rsi(cls, ambiguous='split')    # doctest: +SKIP
    >>> res.rsi                                      # doctest: +SKIP
    array([ 1.0, -1.0,  0.0])
    """
    if ambiguous not in AMBIGUITY_POLICIES:
        raise ValueError(
            f'ambiguous must be one of {AMBIGUITY_POLICIES}, got {ambiguous!r}'
        )
    if substrate_scope not in SUBSTRATE_SCOPES:
        raise ValueError(
            f'substrate_scope must be one of {SUBSTRATE_SCOPES}, '
            f'got {substrate_scope!r}'
        )

    # Products are attributable only where an aligned sequence retains the
    # unmutated substrate, i.e. inside a RIP column.
    prod_fwd = cls.prod_fwd
    prod_rev = cls.prod_rev

    if substrate_scope == 'all':
        sub_fwd_mask, sub_rev_mask = cls.ca, cls.tg
    elif substrate_scope == 'assessable':
        sub_fwd_mask, sub_rev_mask = cls.sub_fwd, cls.sub_rev
    else:  # 'rip_like_columns'
        sub_fwd_mask = cls.ca & cls.fwd_col
        sub_rev_mask = cls.tg & cls.rev_col

    fwd_sub = sub_fwd_mask.sum(axis=1).astype(float)
    rev_sub = sub_rev_mask.sum(axis=1).astype(float)

    # A forward product at column i is also a reverse product when its partner
    # column j carries reverse RIP evidence. Indexing with the clamped partner
    # index is safe because prod_fwd is False wherever no neighbour exists.
    nxt = np.where(cls.next_idx >= 0, cls.next_idx, 0)
    prv = np.where(cls.prev_idx >= 0, cls.prev_idx, 0)

    amb_at_t = prod_fwd & cls.rev_col[nxt]  # ambiguous, indexed at the T column
    amb_at_a = prod_rev & cls.fwd_col[prv]  # the same events, at the A column

    only_fwd = prod_fwd & ~amb_at_t
    only_rev = prod_rev & ~amb_at_a

    fwd_prod = only_fwd.sum(axis=1).astype(float)
    rev_prod = only_rev.sum(axis=1).astype(float)
    n_ambiguous = amb_at_t.sum(axis=1)

    if ambiguous == 'split':
        fwd_prod += 0.5 * n_ambiguous
        rev_prod += 0.5 * n_ambiguous
    elif ambiguous == 'both':
        fwd_prod += n_ambiguous
        rev_prod += n_ambiguous
    elif ambiguous == 'weight':
        # Evidence for the forward strand is the surviving unmutated C at the
        # T's column; for the reverse strand, the surviving G at the A's column.
        nC_i = np.broadcast_to(cls.nC, cls.arr.shape)
        nG_j = cls.nG[nxt]
        den = nC_i + nG_j
        w = np.divide(nC_i, den, out=np.full(cls.arr.shape, 0.5), where=den > 0)
        fwd_prod += (amb_at_t * w).sum(axis=1)
        rev_prod += (amb_at_t * (1.0 - w)).sum(axis=1)
    # 'exclude' adds nothing

    den_f = fwd_prod + fwd_sub
    den_r = rev_prod + rev_sub

    n_rows = cls.arr.shape[0]
    p_fwd = np.full(n_rows, np.nan)
    p_rev = np.full(n_rows, np.nan)
    np.divide(fwd_prod, den_f, out=p_fwd, where=den_f > 0)
    np.divide(rev_prod, den_r, out=p_rev, where=den_r > 0)

    rsi = p_fwd - p_rev

    z, pvalue = _two_proportion_test(fwd_prod, den_f, rev_prod, den_r)

    return RSIResult(
        rsi=rsi,
        p_fwd=p_fwd,
        p_rev=p_rev,
        fwd_prod=fwd_prod,
        fwd_sub=fwd_sub,
        rev_prod=rev_prod,
        rev_sub=rev_sub,
        n_ambiguous=n_ambiguous,
        z=z,
        pvalue=pvalue,
        ambiguous=ambiguous,
        substrate_scope=substrate_scope,
    )
