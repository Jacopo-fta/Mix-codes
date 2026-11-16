import numpy as np
from scipy.special import ellipk, ellipe


def biot_savart_loops_vec(R_src, zc_src, I_src, r_obs, z_obs, mu0=4e-7 * np.pi):
    """
    BIOT_SAVART_LOOPS_VEC  Campo magnetico di N spire coassiali in P punti.

    Calcolo VETTORIZZATO con integrali ellittici completi.
    Le spire sono coassiali con l'asse z, raggio R_src, quota zc_src,
    corrente I_src.

    INPUT:
        R_src, zc_src, I_src : vettori (Ns,) - sorgenti
        r_obs, z_obs         : vettori (Np,) - punti di osservazione
        mu0                  : permeabilita' del vuoto [H/m]

    OUTPUT:
        Br_tot, Bz_tot       : vettori (Np,) - campo totale dovuto a tutte le spire
    """
    R = np.asarray(R_src, dtype=float).ravel()[np.newaxis, :]     # (1, Ns)
    Zc = np.asarray(zc_src, dtype=float).ravel()[np.newaxis, :]   # (1, Ns)
    I = np.asarray(I_src, dtype=float).ravel()[np.newaxis, :]     # (1, Ns)
    rP = np.asarray(r_obs, dtype=float).ravel()[:, np.newaxis]    # (Np, 1)
    zP = np.asarray(z_obs, dtype=float).ravel()[:, np.newaxis]    # (Np, 1)

    N_obs = rP.shape[0]
    N_src = R.shape[1]

    dz = zP - Zc                                  # (Np, Ns)
    Rmat = np.broadcast_to(R, (N_obs, N_src))
    rPmat = np.broadcast_to(rP, (N_obs, N_src))
    Imat = np.broadcast_to(I, (N_obs, N_src))

    alpha2 = Rmat ** 2 + rPmat ** 2 + dz ** 2 - 2.0 * Rmat * rPmat
    beta2 = Rmat ** 2 + rPmat ** 2 + dz ** 2 + 2.0 * Rmat * rPmat
    k2 = 1.0 - alpha2 / beta2

    # Clamp numerico per evitare instabilita' di ellipke
    k2 = np.clip(k2, 0.0, 1.0 - 1e-14)
    K = ellipk(k2)
    E = ellipe(k2)

    Bz_mat = (
        mu0 * Imat / (2.0 * np.pi) / np.sqrt(beta2)
        * (K + E * (Rmat ** 2 - rPmat ** 2 - dz ** 2) / alpha2)
    )

    Br_mat = np.zeros((N_obs, N_src), dtype=float)
    mask = rPmat > 1e-10  # evita divisione per 0 sull'asse
    if np.any(mask):
        Br_mat[mask] = (
            mu0 * Imat[mask] / (2.0 * np.pi) * dz[mask]
            / (rPmat[mask] * np.sqrt(beta2[mask]))
            * (
                -K[mask]
                + E[mask]
                * (Rmat[mask] ** 2 + rPmat[mask] ** 2 + dz[mask] ** 2)
                / alpha2[mask]
            )
        )

    # Sostituzione esatta sull'asse (formula chiusa per Bz, Br=0 per simmetria)
    if np.any(~mask):
        on_axis = ~mask
        Bz_axis = mu0 * Imat * Rmat ** 2 / (2.0 * (Rmat ** 2 + dz ** 2) ** 1.5)
        Bz_mat[on_axis] = Bz_axis[on_axis]

    Br_tot = np.sum(Br_mat, axis=1)
    Bz_tot = np.sum(Bz_mat, axis=1)
    return Br_tot, Bz_tot


if __name__ == "__main__":
    Br, Bz = biot_savart_loops_vec([0.5], [0.0], [50000.0], [0.0, 0.25], [0.0, 0.1])
    print("Br:", Br)
    print("Bz:", Bz)
