import time
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cprintf import cprintf
from assignFixValue import assignFixValue
from factorizeMatrix import factorizeMatrix
from forwardBackwardSolve import forwardBackwardSolve
from iterativeRefinement import iterativeRefinement
from assembleSolution import assembleSolution
from cond1est import cond1est


def directSolve(A, b, idFix=None, xFix=None, *varargin, **kwargs):
    """
    DIRECTSOLVE solves the problem Ax = b by removing from the system the unknowns
    assigned on the boundary.

    USE:
        x, factor = directSolve(A, b, idFix, xFix, *varargin)
        x, factor, c = directSolve(A, b, idFix, xFix, *varargin, return_cond=True)
    """
    method = ""
    factor = None
    iterMAX = 0
    verbose = "y"
    equivalue = []
    binary = None
    return_cond = kwargs.pop("return_cond", False)

    for i in range(0, len(varargin), 2):
        vin = str(varargin[i]).lower()
        val = varargin[i + 1]
        if vin == "method":
            method = str(val)
        elif vin == "factor":
            factor = val
        elif vin == "itermax":
            iterMAX = int(val)
        elif vin == "verbose":
            verbose = str(val)
        elif vin == "equivalue":
            equivalue = val
        elif vin == "binary":
            binary = val

    if "method" in kwargs:
        method = str(kwargs["method"])
    if "factor" in kwargs:
        factor = kwargs["factor"]
    if "iterMAX" in kwargs:
        iterMAX = int(kwargs["iterMAX"])
    if "verbose" in kwargs:
        verbose = str(kwargs["verbose"])
    if "equivalue" in kwargs:
        equivalue = kwargs["equivalue"]
    if "binary" in kwargs:
        binary = kwargs["binary"]

    iprn = verbose.lower() == "y"
    cprintf(iprn, "solving system...\n")
    t1 = time.perf_counter()

    iSolve = True
    n_total = A.shape[1]
    if b is None or np.size(b) == 0:
        b_in = np.zeros((n_total, 1), dtype=float)
        iSolve = False
        b_1d = False
    else:
        b_in = np.asarray(b, dtype=float)
        b_1d = b_in.ndim == 1

    # Convert 1-based idFix if equal to n_total (MATLAB compatibility)
    if idFix is not None and np.size(idFix) > 0:
        idFix_arr = np.asarray(idFix, dtype=int).ravel()
        if np.any(idFix_arr == n_total):
            idFix_arr = idFix_arr - 1
    else:
        idFix_arr = np.array([], dtype=int)

    xFix_arr = np.asarray(xFix, dtype=float) if (xFix is not None and np.size(xFix) > 0) else np.array([], dtype=float)

    # Reduction of solving system
    A_red, b_red, idx = assignFixValue(
        A, b_in, idFix_arr, xFix_arr, "equivalue", equivalue, "binary", binary
    )

    if method.lower() in ("mldivide", "\\"):
        cprintf(iprn, "* MLDIVIDE... ")
        t2 = time.perf_counter()
        if sp.issparse(A_red):
            x0 = spla.spsolve(A_red.tocsc(), b_red)
        else:
            x0 = np.linalg.solve(A_red, b_red)
        factor = SimpleNamespace(x=x0, method=method)
        cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t2)
    else:
        if factor is None or (isinstance(factor, dict) and len(factor) == 0):
            factor = factorizeMatrix(A_red, "method", method, "verbose", verbose)

        if iSolve:
            x0 = forwardBackwardSolve(factor, b_red, "verbose", verbose)
            if iterMAX > 0:
                x0 = iterativeRefinement(
                    A_red, b_red, x0, factor, iterMAX, "verbose", verbose
                )
        else:
            x0 = np.zeros_like(b_red)

    full_dim = (n_total,) if b_1d else b_in.shape
    x = assembleSolution(
        x0, full_dim, idx, idFix_arr, xFix_arr, "equivalue", equivalue, "binary", binary
    )

    cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t1)

    if return_cond:
        c = cond1est(A_red, factor) if method.lower() not in ("mldivide", "\\") else np.linalg.cond(A_red.toarray() if sp.issparse(A_red) else A_red, 1)
        return x, factor, c

    return x, factor
