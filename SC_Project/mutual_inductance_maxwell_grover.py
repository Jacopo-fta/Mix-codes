import numpy as np
from scipy.special import ellipk, ellipe


def mutual_inductance_maxwell_grover(R1, R2, d, mu0=4e-7 * np.pi):
    """
    MUTUAL_INDUCTANCE_MAXWELL_GROVER  Mutua induttanza di due spire coassiali.

    Formula analitica esatta (Maxwell, Grover) basata su integrali ellittici:
        M = mu0 * sqrt(R1*R2) * ((2/k - k)*K(k) - (2/k)*E(k))
        k^2 = 4*R1*R2 / ((R1+R2)^2 + d^2)

    INPUT:
        R1, R2 : raggi delle spire [m]
        d      : distanza assiale tra le spire [m]
        mu0    : permeabilita' del vuoto [H/m]

    OUTPUT:
        M      : mutua induttanza [H]
    """
    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    d = np.asarray(d, dtype=float)

    k2 = 4.0 * R1 * R2 / ((R1 + R2) ** 2 + d ** 2)
    k = np.sqrt(k2)
    K = ellipk(k2)
    E = ellipe(k2)
    M = mu0 * np.sqrt(R1 * R2) * ((2.0 / k - k) * K - (2.0 / k) * E)
    return M


if __name__ == "__main__":
    M_test = mutual_inductance_maxwell_grover(0.5, 0.5, 0.02)
    print(f"Mutua induttanza test (R1=0.5m, R2=0.5m, d=0.02m): {M_test*1e6:.3f} uH")
