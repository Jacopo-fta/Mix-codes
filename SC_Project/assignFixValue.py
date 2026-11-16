import numpy as np
import scipy.sparse as sp


def _parse_varargin(varargin):
    equivalue = []
    binary = np.empty((0, 5), dtype=float)
    for i in range(0, len(varargin), 2):
        vin = str(varargin[i]).lower()
        if vin == "equivalue":
            equivalue = varargin[i + 1] if varargin[i + 1] is not None else []
        elif vin == "binary":
            val = varargin[i + 1]
            if val is not None and np.size(val) > 0:
                binary = np.atleast_2d(np.asarray(val, dtype=float))
    return equivalue, binary


def assignFixValue(A, b, idFix=None, xfix=None, *varargin, **kwargs):
    """
    ASSIGNFIXVALUE assigns to certain idFix a prefixed xfix and rearranges
    the solving matrix and rhs taking into account equivalues and binary constraints.

    NOTE:
        Indices in idFix, equivalue, and binary can be 0-based (Python standard).
        If 1-based indices matching N == A.shape[0] are passed (e.g. N_nodi from MATLAB),
        they are automatically converted to 0-based.
    """
    equivalue, binary = _parse_varargin(varargin)
    if "equivalue" in kwargs and kwargs["equivalue"] is not None:
        equivalue = kwargs["equivalue"]
    if "binary" in kwargs and kwargs["binary"] is not None and np.size(kwargs["binary"]) > 0:
        binary = np.atleast_2d(np.asarray(kwargs["binary"], dtype=float))

    is_sp = sp.issparse(A)
    A_mod = A.tolil().copy() if is_sp else np.array(A, dtype=float, copy=True)

    b_arr = np.array(b, dtype=float, copy=True)
    b_1d = b_arr.ndim == 1
    if b_1d:
        b_arr = b_arr[:, np.newaxis]

    n_total = b_arr.shape[0]

    if idFix is not None and np.size(idFix) > 0:
        idFix_arr = np.asarray(idFix, dtype=int).ravel()
        # Automatic 1-based to 0-based conversion if index == n_total
        if np.any(idFix_arr == n_total):
            idFix_arr = idFix_arr - 1
        idFix_uniq, iElm = np.unique(idFix_arr, return_index=True)
    else:
        idFix_uniq = np.array([], dtype=int)
        iElm = np.array([], dtype=int)

    if idFix_uniq.size > 0 and xfix is not None and np.size(xfix) > 0:
        xfix_arr = np.atleast_1d(np.asarray(xfix, dtype=float))
        if xfix_arr.ndim == 1:
            xfix_arr = xfix_arr[:, np.newaxis]
        if xfix_arr.shape[0] == 1 and idFix_uniq.size == 1:
            xfix_sel = xfix_arr
        else:
            xfix_sel = xfix_arr[iElm, :]
        b_arr = b_arr - A_mod[:, idFix_uniq] @ xfix_sel

    idRem = []

    # equipotential unknowns
    if equivalue:
        for eq in equivalue:
            idEq = np.asarray(eq, dtype=int).ravel()
            if idEq.size > 1:
                A_mod[:, idEq[0]] = np.sum(A_mod[:, idEq], axis=1)
                idRem.extend(idEq[1:].tolist())
                A_mod[idEq[0], :] = np.sum(A_mod[idEq, :], axis=0)
                b_arr[idEq[0], :] = np.sum(b_arr[idEq, :], axis=0)

    # binary constraints
    if binary.shape[0] > 0:
        i1 = np.abs(binary[:, 0]).astype(int)
        i2 = np.abs(binary[:, 1]).astype(int)
        alpha = binary[:, 2]
        beta = binary[:, 3]
        gamma = binary[:, 4]
        for i in range(len(i1)):
            A_mod[:, i1[i]] = A_mod[:, i1[i]] - alpha[i] * A_mod[:, i2[i]] / beta[i]
            b_arr = b_arr - gamma[i] * (A_mod[:, i2[i]] @ np.ones((1, b_arr.shape[1]))) / beta[i]
            A_mod[i1[i], :] = A_mod[i1[i], :] - alpha[i] * A_mod[i2[i], :] / beta[i]
            b_arr[i1[i], :] = b_arr[i1[i], :] - alpha[i] * b_arr[i2[i], :] / beta[i]
    else:
        i2 = np.array([], dtype=int)

    rem_all = np.concatenate(
        [
            idFix_uniq,
            np.asarray(idRem, dtype=int),
            np.asarray(i2, dtype=int),
        ]
    )
    idUnk = np.setdiff1d(np.arange(n_total, dtype=int), rem_all)

    if is_sp:
        A_csr = A_mod.tocsr()
        A_red = A_csr[idUnk, :][:, idUnk]
    else:
        A_red = A_mod[np.ix_(idUnk, idUnk)]

    b_red = b_arr[idUnk, :]
    if b_1d:
        b_red = b_red.ravel()

    return A_red, b_red, idUnk
