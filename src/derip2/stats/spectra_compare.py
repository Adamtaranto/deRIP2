"""
Statistical comparison of mutation spectra.

Two spectra (e.g. two species, two clades, or two precalculated SBS matrices) can
be compared in two complementary ways:

- **Cosine similarity** — a scale-free *effect size* in ``[0, 1]``. 1.0 means the
  two channel profiles have identical shape; lower values mean they differ. It
  ignores the total number of events, so it answers "do these look alike?".
- **Chi-squared test of homogeneity** — a *significance test* of whether the
  channel counts could have come from one shared distribution. It answers "is the
  difference more than sampling noise?", and its per-channel standardised
  residuals show *which* channels drive any difference.

The two are used together: with very large spectra almost any difference is
"significant", so read the p-value alongside the cosine similarity (how different)
and the residuals (different where).

The chi-squared p-value is computed here from the regularised incomplete gamma
function, so this module needs no SciPy — consistent with the rest of deRIP2's
hand-rolled statistics.
"""

import logging
import math
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity between two spectra vectors.

    Parameters
    ----------
    a, b : sequence of float
        Channel count (or proportion) vectors of equal length.

    Returns
    -------
    float
        Cosine similarity in ``[0, 1]`` for non-negative inputs; ``nan`` if either
        vector is all zero.

    Raises
    ------
    ValueError
        If the two vectors have different lengths.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f'Vectors differ in length: {x.shape} vs {y.shape}')
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return float('nan')
    return float(np.dot(x, y) / (nx * ny))


def _gser(a: float, x: float) -> float:
    """
    Lower regularised incomplete gamma ``P(a, x)`` by series expansion.

    Parameters
    ----------
    a : float
        Shape parameter (> 0).
    x : float
        Evaluation point (``0 <= x < a + 1`` for good convergence).

    Returns
    -------
    float
        ``P(a, x)``.
    """
    gln = math.lgamma(a)
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - gln)


def _gcf(a: float, x: float) -> float:
    """
    Upper regularised incomplete gamma ``Q(a, x)`` by continued fraction.

    Parameters
    ----------
    a : float
        Shape parameter (> 0).
    x : float
        Evaluation point (``x >= a + 1`` for good convergence).

    Returns
    -------
    float
        ``Q(a, x)``.
    """
    gln = math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x: float, df: float) -> float:
    """
    Survival function (upper tail) of the chi-squared distribution.

    Parameters
    ----------
    x : float
        Chi-squared statistic (>= 0).
    df : float
        Degrees of freedom (> 0).

    Returns
    -------
    float
        ``P(X > x)`` for a chi-squared variable with ``df`` degrees of freedom.
    """
    if df <= 0:
        return float('nan')
    if x <= 0:
        return 1.0
    a = df / 2.0
    y = x / 2.0
    # Q(a, y) is the upper tail = survival function.
    if y < a + 1.0:
        return 1.0 - _gser(a, y)
    return _gcf(a, y)


def chi2_homogeneity(
    matrix: np.ndarray, sample_names: Optional[List[str]] = None
) -> Dict:
    """
    Chi-squared test of homogeneity across the columns of a count matrix.

    The null hypothesis is that every column (sample/group) draws its channel
    counts from the same underlying distribution. Channels that are empty across
    all samples are dropped and the degrees of freedom reduced accordingly.

    Parameters
    ----------
    matrix : numpy.ndarray
        ``(n_channels, n_samples)`` non-negative count matrix.
    sample_names : list of str, optional
        Column labels, used only for the returned summary.

    Returns
    -------
    dict
        ``chi2`` (statistic), ``dof``, ``pvalue``, ``cramers_v`` (effect size in
        ``[0, 1]``), ``n_samples``, ``n_channels_tested``, ``residuals`` (the
        ``(n_channels, n_samples)`` standardised Pearson residuals, ``0`` for
        dropped channels) and ``sample_names``.

    Raises
    ------
    ValueError
        If the matrix has fewer than two columns or contains negative counts.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] < 2:
        raise ValueError('Need a 2-D matrix with at least two sample columns')
    if (mat < 0).any():
        raise ValueError('Counts must be non-negative')

    n_channels, n_samples = mat.shape
    row_tot = mat.sum(axis=1)
    col_tot = mat.sum(axis=0)
    grand = mat.sum()

    residuals = np.zeros_like(mat)
    if grand == 0:
        return {
            'chi2': 0.0,
            'dof': 0,
            'pvalue': float('nan'),
            'cramers_v': float('nan'),
            'n_samples': n_samples,
            'n_channels_tested': 0,
            'residuals': residuals,
            'sample_names': list(sample_names) if sample_names else None,
        }

    active = row_tot > 0
    expected = np.outer(row_tot, col_tot) / grand
    diff = mat - expected
    with np.errstate(divide='ignore', invalid='ignore'):
        cell_chi2 = np.where(expected > 0, diff**2 / expected, 0.0)
        residuals = np.where(expected > 0, diff / np.sqrt(expected), 0.0)
    chi2 = float(cell_chi2.sum())

    n_active = int(active.sum())
    dof = (n_active - 1) * (n_samples - 1)
    pvalue = chi2_sf(chi2, dof) if dof > 0 else float('nan')
    denom = grand * max(min(n_active - 1, n_samples - 1), 1)
    cramers_v = float(math.sqrt(chi2 / denom)) if denom > 0 else float('nan')

    logger.debug(
        'chi2 homogeneity: chi2=%.3f dof=%d p=%.3g cramersV=%.3f',
        chi2,
        dof,
        pvalue,
        cramers_v,
    )
    return {
        'chi2': chi2,
        'dof': dof,
        'pvalue': pvalue,
        'cramers_v': cramers_v,
        'n_samples': n_samples,
        'n_channels_tested': n_active,
        'residuals': residuals,
        'sample_names': list(sample_names) if sample_names else None,
    }


def compare_spectra(
    a: Sequence[float],
    b: Sequence[float],
    channels: Optional[Sequence[str]] = None,
    *,
    top: int = 8,
) -> Dict:
    """
    Compare two spectra: cosine similarity plus a chi-squared homogeneity test.

    Parameters
    ----------
    a, b : sequence of float
        Channel count vectors of equal length (e.g. two group columns, or one
        column from each of two precalculated matrices in the same context).
    channels : sequence of str, optional
        Channel labels, used to report the most differentiating channels.
    top : int, optional
        How many top differentiating channels to return (default: 8).

    Returns
    -------
    dict
        ``cosine_similarity``, the full chi-squared result (``chi2``, ``dof``,
        ``pvalue``, ``cramers_v``), and ``top_channels`` — a list of
        ``{channel, a, b, residual}`` for the channels with the largest absolute
        difference in standardised residual between the two columns, most extreme
        first.

    Raises
    ------
    ValueError
        If the vectors differ in length, or ``channels`` (when given) does not
        match their length.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f'Vectors differ in length: {x.shape} vs {y.shape}')
    if channels is not None and len(channels) != x.size:
        raise ValueError('channels length must match the spectra length')

    matrix = np.column_stack([x, y])
    chi2 = chi2_homogeneity(matrix, sample_names=['a', 'b'])
    cosine = cosine_similarity(x, y)

    top_channels: List[Dict] = []
    if channels is not None:
        # Difference in standardised residual between the two columns per channel.
        resid = chi2['residuals']
        delta = np.abs(resid[:, 0] - resid[:, 1])
        order = np.argsort(delta)[::-1][:top]
        for i in order:
            if delta[i] == 0:
                continue
            top_channels.append(
                {
                    'channel': channels[i],
                    'a': float(x[i]),
                    'b': float(y[i]),
                    'residual': float(resid[i, 0] - resid[i, 1]),
                }
            )

    return {
        'cosine_similarity': cosine,
        'chi2': chi2['chi2'],
        'dof': chi2['dof'],
        'pvalue': chi2['pvalue'],
        'cramers_v': chi2['cramers_v'],
        'top_channels': top_channels,
    }


def _assert_same_context(channels_a: Sequence[str], channels_b: Sequence[str]) -> None:
    """
    Verify two spectra share the same channel set (hence the same context).

    Two matrices are only comparable if their channel labels match exactly in both
    membership and order: a trinucleotide SBS-96 matrix (``A[C>T]G`` labels) and a
    downstream matrix (``[C>T]AG`` labels) describe different quantities and must
    never be compared.

    Parameters
    ----------
    channels_a, channels_b : sequence of str
        The channel-label lists of the two matrices.

    Raises
    ------
    ValueError
        If the two channel lists differ in length or content/order.
    """
    if list(channels_a) != list(channels_b):
        raise ValueError(
            'Cannot compare spectra with different channel sets: the two matrices '
            'use different sequence contexts (e.g. trinucleotide vs downstream) or '
            'channel orderings.'
        )


def compare_matrix_files(
    path_a: str,
    path_b: str,
    *,
    sample_a: int = 0,
    sample_b: int = 0,
    top: int = 8,
) -> Dict:
    """
    Compare two spectra matrix files, guarding that they share a context.

    Each file is read with :func:`derip2.spectra.matrix_io.read_sbs_matrix`; the
    two channel-label lists must match exactly (same membership and order), which
    guarantees the matrices describe the same sequence context. One sample column
    from each file is then compared with :func:`compare_spectra`.

    Parameters
    ----------
    path_a, path_b : str
        Paths to two ``MutationType`` tab-separated matrix files.
    sample_a, sample_b : int, optional
        Which sample column of each file to compare (default: ``0``).
    top : int, optional
        How many top differentiating channels to return (default: ``8``).

    Returns
    -------
    dict
        The :func:`compare_spectra` result (cosine similarity, chi-squared summary
        and top differentiating channels).

    Raises
    ------
    ValueError
        If the two files use different channel sets (contexts), or if a requested
        sample column is out of range.
    """
    from derip2.spectra.matrix_io import read_sbs_matrix

    channels_a, names_a, matrix_a = read_sbs_matrix(path_a)
    channels_b, names_b, matrix_b = read_sbs_matrix(path_b)
    _assert_same_context(channels_a, channels_b)
    if not 0 <= sample_a < matrix_a.shape[1]:
        raise ValueError(f'sample_a index {sample_a} out of range for {path_a}')
    if not 0 <= sample_b < matrix_b.shape[1]:
        raise ValueError(f'sample_b index {sample_b} out of range for {path_b}')
    logger.debug(
        'Comparing %s[%s] vs %s[%s]',
        path_a,
        names_a[sample_a],
        path_b,
        names_b[sample_b],
    )
    return compare_spectra(
        matrix_a[:, sample_a], matrix_b[:, sample_b], channels_a, top=top
    )


def pairwise_compare(
    matrix: np.ndarray,
    sample_names: Sequence[str],
    *,
    correction: str = 'bonferroni',
) -> List[Dict]:
    """
    Compare every pair of sample columns with a chi-squared homogeneity test.

    Parameters
    ----------
    matrix : numpy.ndarray
        ``(n_channels, n_samples)`` count matrix.
    sample_names : sequence of str
        Column labels, one per sample.
    correction : {'bonferroni', 'none'}, optional
        Multiple-testing correction applied to the pairwise p-values (default:
        ``'bonferroni'``).

    Returns
    -------
    list of dict
        One dict per pair with ``a``, ``b``, ``cosine_similarity``, ``chi2``,
        ``dof``, ``pvalue``, ``pvalue_adjusted`` and ``cramers_v``, sorted by
        ascending adjusted p-value.

    Raises
    ------
    ValueError
        If ``sample_names`` length does not match the number of columns.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    names = list(sample_names)
    if len(names) != mat.shape[1]:
        raise ValueError('sample_names length must match the number of columns')

    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    n_tests = max(len(pairs), 1)
    results: List[Dict] = []
    for i, j in pairs:
        res = compare_spectra(mat[:, i], mat[:, j])
        p = res['pvalue']
        if correction == 'bonferroni' and not math.isnan(p):
            p_adj = min(1.0, p * n_tests)
        else:
            p_adj = p
        results.append(
            {
                'a': names[i],
                'b': names[j],
                'cosine_similarity': res['cosine_similarity'],
                'chi2': res['chi2'],
                'dof': res['dof'],
                'pvalue': p,
                'pvalue_adjusted': p_adj,
                'cramers_v': res['cramers_v'],
            }
        )
    results.sort(
        key=lambda r: (
            math.inf if math.isnan(r['pvalue_adjusted']) else r['pvalue_adjusted']
        )
    )
    return results
