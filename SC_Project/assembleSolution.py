import numpy as np


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


def assembleSolution(x0, dim, idx, idFix=None, xFix=None, *varargin, **kwargs):
    """
    ASSEMBLESOLUTION merges the solution with the fixed values, equivalues
    and binary constraints.

    USE:
        x = assembleSolution(x0, dim, idx, idFix, xFix, *varargin)
    """
    equivalue, binary = _parse_varargin(varargin)
    if "equivalue" in kwargs and kwargs["equivalue"] is not None:
        equivalue = kwargs["equivalue"]
    if "binary" in kwargs and kwargs["binary"] is not None and np.size(kwargs["binary"]) > 0:
        binary = np.atleast_2d(np.asarray(kwargs["binary"], dtype=float))

    x0_arr = np.asarray(x0, dtype=float)
    squeeze_out = x0_arr.ndim == 1
    if x0_arr.ndim == 1:
        x0_arr = x0_arr[:, np.newaxis]

    if isinstance(dim, (int, np.integer)):
        full_shape = (int(dim), x0_arr.shape[1])
    else:
        dim_tuple = tuple(dim)
        if len(dim_tuple) == 1:
            full_shape = (int(dim_tuple[0]), x0_arr.shape[1])
        else:
            full_shape = (int(dim_tuple[0]), int(dim_tuple[1]))

    idx_arr = np.asarray(idx, dtype=int).ravel()
    if idFix is not None and np.size(idFix) > 0:
        idFix_arr = np.asarray(idFix, dtype=int).ravel()
    else:
        idFix_arr = np.array([], dtype=int)

    # Ensure full_shape[0] is large enough to hold all indices (handles MATLAB bug in directSolve where size(b) was passed)
    max_id = 0
    if idx_arr.size > 0:
        max_id = max(max_id, int(np.max(idx_arr)) + 1)
    if idFix_arr.size > 0:
        max_id = max(max_id, int(np.max(idFix_arr)) + 1)
    n_rows = max(full_shape[0], max_id)

    x = np.zeros((n_rows, full_shape[1]), dtype=float)

    # assembling complete solution
    if idx_arr.size > 0:
        x[idx_arr, :] = x0_arr

    # add fixed values
    if idFix_arr.size > 0 and xFix is not None and np.size(xFix) > 0:
        xFix_arr = np.atleast_1d(np.asarray(xFix, dtype=float))
        if xFix_arr.ndim == 1:
            xFix_arr = xFix_arr[:, np.newaxis]
        x[idFix_arr, :] = xFix_arr

    # add equivalues
    for eq in equivalue:
        eq_idx = np.asarray(eq, dtype=int).ravel()
        if eq_idx.size > 0:
            x[eq_idx, :] = x[eq_idx[0], :]

    # add binary constraints
    for i in range(binary.shape[0]):
        i1 = int(abs(binary[i, 0]))
        i2 = int(abs(binary[i, 1]))
        k1 = binary[i, 2]
        k2 = binary[i, 3]
        k3 = binary[i, 4]
        x[i2, :] = (k3 - k1 * x[i1, :]) / k2

    if squeeze_out and x.shape[1] == 1:
        return x.ravel()
    return x
