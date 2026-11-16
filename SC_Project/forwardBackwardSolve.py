import time
import numpy as np
import scipy.linalg as la
from cprintf import cprintf


def forwardBackwardSolve(factor, b, *varargin, **kwargs):
    """
    FORWARDBACKWARDSOLVE performs forward and backward substitutions
    (triangular solve) starting from the factorization obtained by factorizeMatrix.
    """
    verbose = "y"
    transpose = "n"

    for i in range(0, len(varargin), 2):
        vin = str(varargin[i]).lower()
        if vin == "verbose":
            verbose = varargin[i + 1]
        elif vin == "transpose":
            transpose = varargin[i + 1]

    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    if "transpose" in kwargs:
        transpose = kwargs["transpose"]

    iprn = str(verbose).lower() == "y"
    is_transp = str(transpose).lower() in ("y", "transp")

    cprintf(iprn, "* back/forward substitution... ")
    t0 = time.perf_counter()

    b_arr = np.asarray(b, dtype=float)

    if factor.method == "diag":
        x = factor.L @ b_arr
    elif factor.issparse and factor.lu_obj is not None:
        trans_flag = "T" if is_transp else "N"
        x = factor.lu_obj.solve(b_arr, trans=trans_flag)
    elif factor.method == "chol":
        # Full Cholesky: L * L^T * x = b
        y = la.solve_triangular(factor.L, b_arr, lower=True)
        x = la.solve_triangular(factor.L.T, y, lower=False)
    elif factor.method == "ldl":
        if is_transp:
            x = la.solve(factor.A_ref.T, b_arr)
        else:
            x = la.solve(factor.A_ref, b_arr)
    elif factor.method == "lu":
        trans_int = 1 if is_transp else 0
        x = la.lu_solve(factor.lu_obj, b_arr, trans=trans_int)
    else:
        x = la.solve(factor.A_ref, b_arr)

    cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t0)
    return x
