import numpy as np
import scipy.sparse as sp
from forwardBackwardSolve import forwardBackwardSolve


def _norm1(A):
    if sp.issparse(A):
        return float(np.max(np.abs(A).sum(axis=0)))
    return float(np.linalg.norm(A, 1))


def cond1est(A, factor):
    """
    COND1EST estimates the 1-norm condition number of A.

    USE:
        c = cond1est(A, factor)
    """
    if factor.L is None or (hasattr(factor.L, "size") and factor.L.size == 0):
        return 0.0
    return _norm1(A) * _norm1est(factor)


def _norm1est(factor):
    """1-norm estimate of inv(A)."""
    n = factor.L.shape[0]
    est = 0.0
    jold = -1
    x = np.ones(n, dtype=float) / n

    for k in range(1, 3):
        if k == 1:
            est = 0.0
            x = np.ones(n, dtype=float) / n
            jold = -1
        else:
            j = int(np.argmax(np.abs(x)))
            if j == jold:
                break
            x = np.zeros(n, dtype=float)
            x[j] = 1.0
            jold = j

        x = forwardBackwardSolve(factor, x, "verbose", "n")
        est_old = est
        est = float(np.sum(np.abs(x)))
        if k > 1 and est <= est_old:
            break

        s = np.ones(n, dtype=float)
        s[x < 0] = -1.0
        x = forwardBackwardSolve(factor, s, "verbose", "n", "transpose", "y")

    return est
