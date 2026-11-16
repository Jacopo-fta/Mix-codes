import time
import numpy as np
import scipy.sparse as sp
from cprintf import cprintf


def checkMatrixStructure(M, *varargin, **kwargs):
    """
    CHECKMATRIXSTRUCTURE checks if matrix is diagonal, symmetric, hermitian, or unsymmetric.

    USE:
        structure = checkMatrixStructure(M, *varargin)
    """
    verbose = "y"
    threshold = 1e-6

    for i in range(0, len(varargin), 2):
        vin = str(varargin[i]).lower()
        if vin == "verbose":
            verbose = varargin[i + 1]
        elif vin == "threshold":
            threshold = float(varargin[i + 1])

    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    if "threshold" in kwargs:
        threshold = float(kwargs["threshold"])

    iprn = str(verbose).lower() == "y"
    cprintf(iprn, "* Structure check... ")
    t0 = time.perf_counter()

    is_sp = sp.issparse(M)
    if is_sp:
        M_csr = M.tocsr()
        d = np.abs(M_csr.diagonal()).copy()
        d[d == 0] = 1.0
        inv_d = sp.diags(1.0 / d, 0, shape=M.shape, format="csr")

        # Check diagonal
        off_diag = M_csr - sp.diags(M_csr.diagonal(), 0, shape=M.shape, format="csr")
        if off_diag.nnz == 0 or np.all(np.abs(off_diag.data) == 0):
            structure = "diagonal"
            cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t0)
            return structure

        # Symmetric check
        dM = M_csr - M_csr.T
        norm_dM = inv_d @ dM
        if norm_dM.nnz == 0 or np.all(np.abs(norm_dM.data) < threshold):
            structure = "symmetric"
            cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t0)
            return structure

        # Hermitian check
        if np.iscomplexobj(M_csr.data):
            dM_h = M_csr - M_csr.conj().T
            norm_dMh = inv_d @ dM_h
            if norm_dMh.nnz == 0 or np.all(np.abs(norm_dMh.data) < threshold):
                structure = "hermitian"
                cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t0)
                return structure
    else:
        M_arr = np.asarray(M)
        d = np.abs(np.diag(M_arr)).copy()
        d[d == 0] = 1.0

        # Check diagonal
        off_diag = M_arr - np.diag(np.diag(M_arr))
        if np.count_nonzero(off_diag) == 0:
            structure = "diagonal"
            cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t0)
            return structure

        # Symmetric check
        dM = (M_arr - M_arr.T) / d[:, np.newaxis]
        nz_vals = dM[dM != 0]
        if nz_vals.size == 0 or np.all(np.abs(nz_vals) < threshold):
            structure = "symmetric"
            cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t0)
            return structure

        # Hermitian check
        if np.iscomplexobj(M_arr):
            dM_h = (M_arr - M_arr.conj().T) / d[:, np.newaxis]
            nz_vals_h = dM_h[dM_h != 0]
            if nz_vals_h.size == 0 or np.all(np.abs(nz_vals_h) < threshold):
                structure = "hermitian"
                cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t0)
                return structure

    structure = "unsymmetric"
    cprintf(iprn, "done %6.2f sec\n", time.perf_counter() - t0)
    return structure
