import numpy as np


def segmentMagneticField1d(P, E, Is, Q):
    """
    SEGMENTMAGNETICFIELD1D evaluates magnetic field in a set of points due to
    thin wire segments in 3D space.

    INPUT:
        P  : point coordinates (Np x 3)
        E  : edge connectivity matrix (Ne x 2) - accepts both 0-based and 1-based indices
        Is : vector with currents (Ne,) or scalar
        Q  : field points (Nf x 3)

    OUTPUT:
        H  : three components of magnetic field (Nf x 3)
    """
    P = np.asarray(P, dtype=float)
    E = np.asarray(E, dtype=int)
    Q = np.asarray(Q, dtype=float)

    # Support both 1-based (MATLAB style) and 0-based indexing automatically
    if E.min() >= 1 and E.max() == P.shape[0]:
        E_idx = E - 1
    else:
        E_idx = E

    # start node and end node
    P1 = P[E_idx[:, 0], :]
    P2 = P[E_idx[:, 1], :]

    nfield = Q.shape[0]
    nsource = E_idx.shape[0]
    H = np.zeros((nfield, 3), dtype=float)

    Is_arr = np.asarray(Is, dtype=float)
    if Is_arr.ndim == 0 or Is_arr.size == 1:
        Is_vec = np.full(nsource, float(Is_arr.ravel()[0]), dtype=float)
    else:
        Is_vec = Is_arr.ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        if nsource < nfield:
            # loop over segments that form current path
            k = np.zeros((nfield, 3), dtype=float)
            for i in range(nsource):
                k[:, 0] = (
                    P1[i, 1] * (P2[i, 2] - Q[:, 2])
                    + P2[i, 1] * (Q[:, 2] - P1[i, 2])
                    + Q[:, 1] * (P1[i, 2] - P2[i, 2])
                )
                k[:, 1] = (
                    -P1[i, 0] * (P2[i, 2] - Q[:, 2])
                    - P2[i, 0] * (Q[:, 2] - P1[i, 2])
                    - Q[:, 0] * (P1[i, 2] - P2[i, 2])
                )
                k[:, 2] = (
                    P1[i, 0] * (P2[i, 1] - Q[:, 1])
                    + P2[i, 0] * (Q[:, 1] - P1[i, 1])
                    + Q[:, 0] * (P1[i, 1] - P2[i, 1])
                )

                a = (
                    (P1[i, 0] - P2[i, 0]) ** 2
                    + (P1[i, 1] - P2[i, 1]) ** 2
                    + (P1[i, 2] - P2[i, 2]) ** 2
                )
                b = (
                    2.0 * (P1[i, 0] - P2[i, 0]) * (P1[i, 0] - Q[:, 0])
                    + 2.0 * (P1[i, 1] - P2[i, 1]) * (P1[i, 1] - Q[:, 1])
                    + 2.0 * (P1[i, 2] - P2[i, 2]) * (P1[i, 2] - Q[:, 2])
                )
                c = (
                    (P1[i, 0] - Q[:, 0]) ** 2
                    + (P1[i, 1] - Q[:, 1]) ** 2
                    + (P1[i, 2] - Q[:, 2]) ** 2
                )

                sqrt_c = np.sqrt(c)
                sqrt_abc = np.sqrt(np.maximum(a - b + c, 0.0))
                denom = sqrt_c * (4.0 * a * c - b ** 2) * sqrt_abc
                g = (
                    (Is_vec[i] / (2.0 * np.pi))
                    * (2.0 * a * sqrt_c - b * sqrt_c + b * sqrt_abc)
                    / denom
                )
                g[~np.isfinite(g)] = 0.0
                H += k * g[:, np.newaxis]
        else:
            # loop over field points
            a = np.sum((P1 - P2) ** 2, axis=1)
            dP = P1 - P2
            k = np.zeros((nsource, 3), dtype=float)
            for i in range(nfield):
                qi = Q[i, :]
                k[:, 0] = (
                    P1[:, 1] * (P2[:, 2] - qi[2])
                    + P2[:, 1] * (qi[2] - P1[:, 2])
                    + qi[1] * dP[:, 2]
                )
                k[:, 1] = -(
                    P1[:, 0] * (P2[:, 2] - qi[2])
                    + P2[:, 0] * (qi[2] - P1[:, 2])
                    + qi[0] * dP[:, 2]
                )
                k[:, 2] = (
                    P1[:, 0] * (P2[:, 1] - qi[1])
                    + P2[:, 0] * (qi[1] - P1[:, 1])
                    + qi[0] * dP[:, 1]
                )

                diff1 = P1 - qi
                b = 2.0 * np.sum(dP * diff1, axis=1)
                c = np.sum(diff1 ** 2, axis=1)

                sqrt_c = np.sqrt(c)
                sqrt_abc = np.sqrt(np.maximum(a - b + c, 0.0))
                denom = sqrt_c * (4.0 * a * c - b ** 2) * sqrt_abc
                g = (
                    (Is_vec / (2.0 * np.pi))
                    * (2.0 * a * sqrt_c - b * sqrt_c + b * sqrt_abc)
                    / denom
                )
                g[~np.isfinite(g)] = 0.0
                H[i, :] = np.sum(k * g[:, np.newaxis], axis=0)

    return H


if __name__ == "__main__":
    # Quick test: square loop in xy plane
    P_test = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
    E_test = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    Q_test = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    H_res = segmentMagneticField1d(P_test, E_test, 100.0, Q_test)
    print("H at center and (0,0,1):\n", H_res)
