import numpy as np
from pyXSteam.XSteam import XSteam

# Inizializzazione tabelle del vapore nel sistema MKS (bar, °C, kJ/kg, m, s)
steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

# %% DATA
P_th = 34.8  # MW
L = 10.0
g = 9.81
eps = 5e-5

k_cooler = 20.0
k_heater = 40.0

# LIQUID (p = 70 bar)
h_l_kj = steamTable.hL_p(70.0)
rho_l = steamTable.rhoL_p(70.0)
mu_l = steamTable.my_ph(70.0, h_l_kj)
h_l = h_l_kj * 1e-3  # MJ/kg

# VAPOR (p = 70 bar)
h_v_kj = steamTable.hV_p(70.0)
rho_v = steamTable.rhoV_p(70.0)
mu_v = steamTable.my_ph(70.0, h_v_kj)
h_v = h_v_kj * 1e-3  # MJ/kg

# %% 1.1c
m_dot = P_th / abs(h_v - h_l)
D = 0.304800609601219

# LIQUID
v_l = m_dot / (rho_l * np.pi * D**2 / 4.0)
Re_l = 4.0 * m_dot / (mu_l * np.pi * D)
f_l = (-1.8 * np.log10((eps / D / 3.7)**1.1 + 6.9 / Re_l))**(-2)

# VAPOR
v_v = m_dot / (rho_v * np.pi * D**2 / 4.0)
Re_v = 4.0 * m_dot / (mu_v * np.pi * D)
f_v = (-1.8 * np.log10((eps / D / 3.7)**1.1 + 6.9 / Re_v))**(-2)

H_L = (k_cooler * 0.5 * rho_l * v_l**2 + k_heater * 0.5 * rho_v * v_v**2) / (
    g * (rho_l - rho_v) - f_l * 0.5 * rho_l * v_l**2 / D - f_v * 0.5 * rho_v * v_v**2 / D
)

# %% Risultati
print("\n--- RISULTATI ASSIGNMENT 1.1c (Rugosità) ---")
print(f"Fattore attrito liquido (f_l): {f_l:8.5f}")
print(f"Fattore attrito vapore (f_v):  {f_v:8.5f}")
print(f"Altezza finale H_L con rugosità: {H_L:6.2f} m")