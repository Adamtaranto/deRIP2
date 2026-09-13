"""
Hand-rolled numerical routines shared by the statistics modules.

deRIP2 keeps NumPy as its only numerical dependency, so the few special
functions the statistics need — an upper regularised incomplete gamma for the
chi-squared survival function, the normal quantile, and a two-proportion
z-test — live here rather than being pulled from SciPy.
"""

import math

import numpy as np

__all__ = ['chi2_sf', 'normal_ppf', 'two_proportion_test']


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


def normal_ppf(p: float) -> float:
    """
    Inverse standard-normal CDF via the Acklam rational approximation.

    Dependency-free (no SciPy), accurate to ~1e-9 over the open interval, which is
    ample for turning a significance level into a critical z-value.

    Parameters
    ----------
    p : float
        Probability in ``(0, 1)``.

    Returns
    -------
    float
        The quantile ``z`` such that ``P(Z <= z) = p`` for ``Z ~ N(0, 1)``.
    """
    # Coefficients for the central and tail regions of Acklam's algorithm.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p <= 0.0:
        return float('-inf')
    if p >= 1.0:
        return float('inf')
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def two_proportion_test(x1, n1, x2, n2):
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
