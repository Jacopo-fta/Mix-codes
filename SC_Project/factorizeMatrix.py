import time
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from cprintf import cprintf
from checkMatrixStructure import checkMatrixStructure


class FactorStruct(SimpleNamespace):
    """Factorization struct compatible with MATLAB factorizeMatrix output."""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def factorizeMatrix(A, *varargin, **kwargs):
    """
    FACTORIZEMATRIX factorizes matrix A using Cholesky, LDL or LU factorization.
    To be used in conjunction with forwardBackwardSolve.
    """
    method = ""
    verbose = "y"

    for i in range(0, len(varargin), 2):
        vin = str(varargin[i]).lower()
        if vin in ("method", "alg"):
            method = str(varargin[i + 1]).lower()
        elif vin == "verbose":
            verbose = varargin[i + 1]

    if "method" in kwargs:
        method = str(kwargs["method"]).lower()
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]

    iprn = str(verbose).lower() == "y"

    if method not in ("chol", "ldl", "lu", ""):
        raise ValueError(f"Algorithm {method} not available")

    is_sp = sp.issparse(A)
    is_real = not np.iscomplexobj(A)

    factor = FactorStruct(
        isreal=is_real,
        issparse=is_sp,
        structure=None,
        method=method,
        L=None,
        D=None,
        U=None,
        p=None,
        q=None,
        S=None,
        R=None,
        spd=False,
        lu_obj=None,
        A_ref=A,
    )

    factor.structure = checkMatrixStructure(A, "threshold", 1e-6, "verbose", verbose)

    # Diagonal matrix
    if factor.structure == "diagonal":
        factor.method = "diag"
        if is_sp:
            d = A.diagonal()
            factor.L = sp.diags(1.0 / d, 0, shape=A.shape, format="csr")
        else:
            factor.L = np.diag(1.0 / np.diag(A))
        return factor

    # Check for approximate positive definiteness
    if factor.structure in ("symmetric", "hermitian"):
        d = A.diagonal() if is_sp else np.diag(A)
        factor.spd = bool(np.all(np.real(d) > 0) and np.all(np.imag(d) == 0))
    else:
        factor.spd = False

    # Automatic algorithm choice
    if not factor.method:
        if factor.spd:
            factor.method = "chol"
        elif factor.structure in ("symmetric", "hermitian") and (factor.isreal or not factor.issparse):
            factor.method = "ldl"
        else:
            factor.method = "lu"

    # Cholesky factorization
    if factor.method == "chol":
        t1 = time.perf_counter()
        if factor.issparse:
            cprintf(iprn, "* CHOL/SPLU factorization of sparse matrix... ")
            try:
                # Attempt Cholesky on dense if small, or use SuperLU for general sparse SPD
                A_csc = A.tocsc()
                factor.lu_obj = spla.splu(A_csc)
                factor.L = A_csc  # non-empty marker for cond1est
                cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t1)
                return factor
            except Exception:
                cprintf(iprn, "failed %6.2f sec\n", time.perf_counter() - t1)
                factor.spd = False
                factor.method = "ldl" if factor.isreal else "lu"
        else:
            cprintf(iprn, "* CHOL factorization of full matrix... ")
            try:
                factor.L = la.cholesky(A, lower=True)
                cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t1)
                return factor
            except la.LinAlgError:
                cprintf(iprn, "failed %6.2f sec\n", time.perf_counter() - t1)
                factor.spd = False
                factor.method = "ldl" if factor.isreal else "lu"

    # LDL factorization
    if factor.method == "ldl":
        t1 = time.perf_counter()
        if factor.issparse:
            cprintf(iprn, "* LDL factorization... ")
            A_csc = A.tocsc()
            factor.lu_obj = spla.splu(A_csc)
            factor.L = A_csc
            cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t1)
            return factor
        else:
            cprintf(iprn, "* LDL factorization of full matrix... ")
            lu, d, perm = la.ldl(A, lower=True)
            factor.L = lu
            factor.D = d
            factor.p = perm
            cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t1)
            return factor

    # LU factorization
    if factor.method == "lu":
        t1 = time.perf_counter()
        if factor.issparse:
            cprintf(iprn, "* LU factorization... ")
            A_csc = A.tocsc()
            factor.lu_obj = spla.splu(A_csc)
            factor.L = A_csc
            cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t1)
        else:
            cprintf(iprn, "* LU factorization of full matrix... ")
            lu_piv = la.lu_factor(A)
            factor.lu_obj = lu_piv
            factor.L = lu_piv[0]
            cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t1)

    return factor
