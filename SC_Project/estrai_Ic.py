import warnings
import numpy as np
import matplotlib.pyplot as plt


def estrai_Ic(filename, L_taps_cm=1.0, Vmin=1e-6, Vmax=1e-4, plot=False):
    """
    ESTRAI_IC  Estrae corrente critica e n-value da una curva I-V (4-probe).

    USO:
        Ic, n, P, Vc = estrai_Ic('IV_1.csv')
        Ic, n, P, Vc = estrai_Ic('IV_1.csv', L_taps_cm=1.0, Vmin=1e-6, Vmax=1e-4, plot=False)

    INPUT:
        filename  : CSV con colonne [Time(s), Current(A), Voltage(V), Temperature(K)]
        L_taps_cm : (default 1.0) distanza tra i tap di tensione [cm]
        Vmin      : (default 1e-6) estremo inferiore finestra di fit [V]
        Vmax      : (default 1e-4) estremo superiore finestra di fit [V]
        plot      : (default False) se True, mostra il fit

    OUTPUT:
        Ic       : corrente critica al criterio Vc = 1 uV/cm * L_taps [A]
        n_value  : esponente power-law transition [-]
        P        : coefficienti polyfit [n_value, q]
        Vc       : criterio in tensione effettivamente usato [V]
    """
    try:
        dati = np.loadtxt(filename, delimiter=",")
    except ValueError:
        dati = np.genfromtxt(filename, delimiter=",", skip_header=1)

    I_raw = dati[:, 1]
    V_raw = dati[:, 2]

    # Criterio di tensione: 1 uV/cm moltiplicato per la distanza tra i tap
    Vc = 1e-6 * L_taps_cm

    # Finestra di fit (zona power-law, prima del flux-flow lineare)
    idx = (V_raw >= Vmin) & (V_raw <= Vmax) & (I_raw > 0)
    I = I_raw[idx]
    V = V_raw[idx]

    if I.size < 5:
        warnings.warn(
            f"Solo {I.size} punti nella finestra di fit per {filename}",
            UserWarning,
        )

    # Fit log-log: log(V) = n*log(I) + q
    P = np.polyfit(np.log(I), np.log(V), 1)
    n_value = P[0]
    q = P[1]
    Ic = np.exp((np.log(Vc) - q) / n_value)

    if plot:
        plt.figure(num=f"Fit {filename}")
        plt.semilogy(I_raw, V_raw, "b.", label="Dati grezzi")
        plt.semilogy(
            I,
            np.exp(np.polyval(P, np.log(I))),
            "r-",
            linewidth=2,
            label="Fit power-law",
        )
        plt.axhline(Vc, color="k", linestyle="--", label=f"Vc = {Vc*1e6:.1f} uV")
        plt.axvline(Ic, color="g", linestyle="--", label=f"Ic = {Ic:.2f} A")
        plt.grid(True)
        plt.xlabel("Corrente [A]")
        plt.ylabel("Tensione [V]")
        plt.title(f"{filename}: Ic = {Ic:.2f} A, n = {n_value:.2f}")
        plt.legend(loc="best")

    return Ic, n_value, P, Vc
