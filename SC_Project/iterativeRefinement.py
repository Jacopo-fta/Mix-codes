import time
import numpy as np
from cprintf import cprintf
from forwardBackwardSolve import forwardBackwardSolve


def iterativeRefinement(A, b, x, factor, iterMAX, *varargin, **kwargs):
    """
    ITERATIVEREFINEMENT applies iterative refinement on the solution x of
    Ax = b given the factorization.
    """
    verbose = "y"
    for i in range(0, len(varargin), 2):
        vin = str(varargin[i]).lower()
        if vin == "verbose":
            verbose = varargin[i + 1]
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]

    iprn = str(verbose).lower() == "y"
    cprintf(iprn, "* iterative refinement... ")
    t1 = time.perf_counter()

    x_curr = np.array(x, dtype=float, copy=True)
    for _ in range(int(iterMAX)):
        r = b - A @ x_curr
        e = forwardBackwardSolve(factor, r, "verbose", "n")
        x_curr = x_curr + e

    cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t1)
    return x_curr
