import numpy as np


def Mu0():
    """
    MU0 gives the value of vacuum magnetic permeability.

    OUTPUT:
        value: 4e-7 * pi
    """
    return 4e-7 * np.pi


if __name__ == "__main__":
    print(f"Mu0 = {Mu0():.6e} H/m")
