import numpy as np
import pandas as pd

# Parametri geometrici VVER-440 (IAEA TRS 379, Table IV)
N_rods = 126
h_stack = 242.0  # cm (2.42 m)
d_out = 0.757    # cm (7.57 mm)

# Frazione massica di U in UO2 (arricchimento ~4%)
e = 0.04
MM_U = 1.0 / (e / 235.0439 + (1.0 - e) / 238.0508)
f_U = MM_U / (MM_U + 2 * 15.999)  # ~0.8815
print(f' La massa dell uranio è {MM_U}')
# Vettori su cui iterare
d_in_list = [0.14, 0.15, 0.16]               # cm (1.4, 1.5, 1.6 mm)
rho_list = [10.4, 10.55, 10.6, 10.7, 10.8]   # g/cm^3

# Costruzione tabella Massa Totale di Uranio nell'Assembly [kg]
results = {}
for rho in rho_list:
    masses = []
    for d_in in d_in_list:
        A_pellet = (np.pi / 4.0) * (d_out**2 - d_in**2)
        V_assembly = A_pellet * h_stack * N_rods
        M_U_assembly = (V_assembly * rho / 1000.0) * f_U
        masses.append(round(M_U_assembly, 2))
    results[f"rho = {rho} g/cm3"] = masses

df = pd.DataFrame(results, index=[f"d_in = {d*10:.1f} mm" for d in d_in_list])
print("Massa totale di Uranio nell'Assembly [kg] (Rif. IAEA Table IV: 119.7 kg):")
print(df)

