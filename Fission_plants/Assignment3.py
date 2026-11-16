import numpy as np
from scipy.io import loadmat
from pyXSteam.XSteam import XSteam

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

# %% Data
NP = 3400e6                      # [W] Nominal power
perc_f = 0.974
power_fuel = NP * perc_f         # [W]

p_nom = 155.13                   # [bar]
p_min = 151.0                    # [bar]

# %% Geometric parameters
H = 168.0 * 2.54e-2              # [m] Active length
D_pellet = 0.3225 * 2.54e-2      # [m] Fuel pellet diameter
L_f = 0.387 * 2.54e-2            # [m] Fuel pellet length
HCF = 2.60

n_assembly = 157
rods_UO2 = 264
perc_bypass = 0.059
A_f = 3.8833                     # [m^2]
m_active = 106.8e6 * 0.453592 / 3600.0  # [kg/s]

D_core = 119.7 * 2.54e-2         # [m]
D_co = 0.374 * 2.54e-2           # [m] Outer cladding diameter
t_cl = 0.0225 * 2.54e-2          # [m] Cladding thickness
D_ci = D_co - 2.0 * t_cl         # [m] Inner cladding diameter
A_ht = 5267.6                    # [m^2]

# %% Zircaloy-4 data
sigma_y = 241.0                  # [MPa] Yield strength
sigma_u = 413.0                  # [MPa] Ultimate tensile strength

# %% End-of-life and gas data
BU = 60.0                        # [MWd/kgU] Burnup
C_N2 = 25e-6                     # [-] Nitrogen impurity
C_H2O = 75e-6                    # [-] Water impurity
Y = 0.28                         # [-] Fission yield
R_r = 0.40                       # [-] Release rate of Xe and Kr

# %% Inputs from Assignment 2
T_ci = loadmat('Tci_data.mat')['T_ci'].flatten()
T_co = loadmat('Tco_data.mat')['T_co'].flatten()

if len(T_ci) != len(T_co):
    raise ValueError("T_ci and T_co must have the same number of axial points.")

N_z = len(T_ci)
z = np.linspace(0.0, H, N_z)

# %% Point 1: Preliminary buckling verification
r_avg = ((D_co / 2.0) + (D_ci / 2.0)) / 2.0

E = lambda T: (9.9e3 - 5.669 * ((T + 273.15) - 273.0)) * 9.81
nu = lambda T: 0.3303 + 8.376e-5 * ((T + 273.15) - 273.0)

T_cl_ave = (T_co + T_ci) / 2.0
p_cr = (E(T_cl_ave) / (4.0 * (1.0 - nu(T_cl_ave)**2))) * (t_cl / r_avg)**3  # [MPa]
sigma_max = np.max(p_cr)

S_m = min((2.0 / 3.0) * sigma_y, (1.0 / 3.0) * sigma_u)
sigma_a = S_m
sigma_a_sec = 3.0 * S_m

if np.min(p_cr * 10.0) > p_nom:
    print("The critical pressure for buckling exceeds nominal pressure (Safety condition)")
else:
    print("The critical pressure for buckling does not exceed nominal pressure (Warning condition)")

print(f"Minimum critical buckling pressure: {np.min(p_cr) * 10.0:.4f} bar")

# %% Point 2: Maximum internal pressure
pi_max = sigma_y * (t_cl / r_avg)  # [MPa]

# %% Point 3: Mechanical stresses
p_i = pi_max
p_0 = p_nom * 0.1

r_co = D_co / 2.0
r_ci = D_ci / 2.0

# Inner surface mechanical stresses
sigma_hin = (p_i * (r_co**2 + r_ci**2) - 2.0 * p_0 * r_co**2) / (r_co**2 - r_ci**2)
sigma_rin = -p_i
sigma_lin = r_ci**2 * (p_i - p_0) / (r_co**2 - r_ci**2)

# Outer surface mechanical stresses
sigma_hout = (2.0 * p_i * r_ci**2 - p_0 * (r_co**2 + r_ci**2)) / (r_co**2 - r_ci**2)
sigma_rout = -p_0
sigma_lout = r_ci**2 * (p_i - p_0) / (r_co**2 - r_ci**2)

# %% Point 4: Thermal stresses
alpha = lambda T: (6.72e-6 * (T + 273.15) - 2.07e-3) / ((T + 273.15) - 308.0)

sigma_rth_in = np.zeros(N_z)
sigma_hth_in = (
    (E(T_ci) * alpha(T_ci) * (T_ci - T_co)) / (2.0 * (1.0 - nu(T_ci)))
    * (1.0 - (2.0 * r_co**2 / (r_co**2 - r_ci**2)) * np.log(r_co / r_ci))
    / np.log(r_co / r_ci)
)
sigma_lth_in = sigma_hth_in.copy()

sigma_rth_out = np.zeros(N_z)
sigma_hth_out = (
    (E(T_co) * alpha(T_co) * (T_ci - T_co)) / (2.0 * (1.0 - nu(T_co)))
    * (1.0 - (2.0 * r_ci**2 / (r_co**2 - r_ci**2)) * np.log(r_co / r_ci))
    / np.log(r_co / r_ci)
)
sigma_lth_out = sigma_hth_out.copy()

# %% Point 5: ASME verification
P_m_h = (sigma_hout + sigma_hin) / 2.0
P_m_r = (sigma_rout + sigma_rin) / 2.0
P_m_l = (sigma_lout + sigma_lin) / 2.0

P_m_h_z = P_m_h * np.ones(N_z)
P_m_r_z = P_m_r * np.ones(N_z)
P_m_l_z = P_m_l * np.ones(N_z)

Q_h_in = sigma_hth_in + sigma_hin - P_m_h
Q_h_out = sigma_hth_out + sigma_hout - P_m_h

Q_r_in = sigma_rth_in + sigma_rin - P_m_r
Q_r_out = sigma_rth_out + sigma_rout - P_m_r

Q_l_in = sigma_lth_in + sigma_lin - P_m_l
Q_l_out = sigma_lth_out + sigma_lout - P_m_l

T_h_in = P_m_h_z + Q_h_in
T_r_in = P_m_r_z + Q_r_in
T_l_in = P_m_l_z + Q_l_in

T_h_out = P_m_h_z + Q_h_out
T_r_out = P_m_r_z + Q_r_out
T_l_out = P_m_l_z + Q_l_out

# %% Tresca criterion
sigma_Tresca_in = max(abs(sigma_hin - sigma_rin), abs(sigma_rin - sigma_lin), abs(sigma_hin - sigma_lin))
sigma_Tresca_out = max(abs(sigma_hout - sigma_rout), abs(sigma_rout - sigma_lout), abs(sigma_hout - sigma_lout))
sigma_max_primary = max(sigma_Tresca_in, sigma_Tresca_out)

sigma_Tresca_in_th = np.maximum.reduce([
    np.abs(sigma_hth_in - sigma_rth_in),
    np.abs(sigma_rth_in - sigma_lth_in),
    np.abs(sigma_hth_in - sigma_lth_in)
])
sigma_Tresca_out_th = np.maximum.reduce([
    np.abs(sigma_hth_out - sigma_rth_out),
    np.abs(sigma_rth_out - sigma_lth_out),
    np.abs(sigma_hth_out - sigma_lth_out)
])

idx_th_in = int(np.argmax(sigma_Tresca_in_th))
sigma_Tresca_in_th_max = sigma_Tresca_in_th[idx_th_in]

idx_th_out = int(np.argmax(sigma_Tresca_out_th))
sigma_Tresca_out_th_max = sigma_Tresca_out_th[idx_th_out]

sigma_tresca_m_max = max(abs(P_m_h - P_m_r), abs(P_m_r - P_m_l), abs(P_m_h - P_m_l))

sigma_tresca_tot_in = np.maximum.reduce([
    np.abs(T_h_in - T_r_in),
    np.abs(T_r_in - T_l_in),
    np.abs(T_h_in - T_l_in)
])
sigma_tresca_tot_out = np.maximum.reduce([
    np.abs(T_h_out - T_r_out),
    np.abs(T_r_out - T_l_out),
    np.abs(T_h_out - T_l_out)
])

idx_crit_in = int(np.argmax(sigma_tresca_tot_in))
sigma_tresca_tot_max_in = sigma_tresca_tot_in[idx_crit_in]

idx_crit_out = int(np.argmax(sigma_tresca_tot_out))
sigma_tresca_tot_max_out = sigma_tresca_tot_out[idx_crit_out]

if sigma_tresca_tot_max_in >= sigma_tresca_tot_max_out:
    sigma_tresca_tot_max = sigma_tresca_tot_max_in
    idx_crit = idx_crit_in
    critical_surface = 'INNER'
    sigma_h_crit = T_h_in[idx_crit]
    sigma_r_crit = T_r_in[idx_crit]
    sigma_l_crit = T_l_in[idx_crit]
else:
    sigma_tresca_tot_max = sigma_tresca_tot_max_out
    idx_crit = idx_crit_out
    critical_surface = 'OUTER'
    sigma_h_crit = T_h_out[idx_crit]
    sigma_r_crit = T_r_out[idx_crit]
    sigma_l_crit = T_l_out[idx_crit]

z_crit = z[idx_crit]

# %% ASME checks & Results
print("\n--- PRIMARY STRESS VERIFICATION ---")
print(f"Primary membrane Tresca stress = {sigma_tresca_m_max:.4f} MPa")
print(f"Allowable primary stress, S_m = {S_m:.4f} MPa")
print(f"Check: {sigma_tresca_m_max:.4f} <= {S_m:.4f} MPa")

if sigma_tresca_m_max <= S_m:
    print("RESULT: VERIFIED")
    print(f"Safety margin: {S_m - sigma_tresca_m_max:.4f} MPa")
    print(f"Utilization ratio: {sigma_tresca_m_max / S_m:.4f}")
else:
    print("RESULT: NOT VERIFIED")
    print(f"Exceedance: {sigma_tresca_m_max - S_m:.4f} MPa")
    print(f"Utilization ratio: {sigma_tresca_m_max / S_m:.4f}")

print("\n--- TRESCA MAXIMA BY SURFACE ---")
print(f"Inner surface: Max total Tresca = {sigma_tresca_tot_max_in:.4f} MPa at z = {z[idx_crit_in]:.6f} m "
      f"(T_ci = {T_ci[idx_crit_in]:.4f} °C, T_co = {T_co[idx_crit_in]:.4f} °C)")
print(f"Outer surface: Max total Tresca = {sigma_tresca_tot_max_out:.4f} MPa at z = {z[idx_crit_out]:.6f} m "
      f"(T_ci = {T_ci[idx_crit_out]:.4f} °C, T_co = {T_co[idx_crit_out]:.4f} °C)")

print("\n--- PRIMARY + SECONDARY STRESS VERIFICATION ---")
print(f"Critical surface: {critical_surface} at z = {z_crit:.4f} m")
print(f"Maximum total Tresca stress = {sigma_tresca_tot_max:.4f} MPa (Allowable 3*S_m = {3.0 * S_m:.4f} MPa)")
if sigma_tresca_tot_max <= 3.0 * S_m:
    print(f"RESULT: VERIFIED (Margin: {3.0 * S_m - sigma_tresca_tot_max:.4f} MPa, Ratio: {sigma_tresca_tot_max / (3.0 * S_m):.4f})")
else:
    print(f"RESULT: NOT VERIFIED (Exceedance: {sigma_tresca_tot_max - 3.0 * S_m:.4f} MPa)")

# %% Point 6: Gas plenum sizing
rho_UO2 = 10960.0
rho_UO2_eff = 0.955 * rho_UO2
M_UO2 = rho_UO2_eff * np.pi * D_pellet**2 / 4.0 * H

u_N2 = 0.028014
u_H2O = 0.018015
n_N2 = C_N2 * M_UO2 / u_N2
n_H2O = C_H2O * M_UO2 / u_H2O

e = 0.0235
u_U = 0.235 * e + 0.238 * (1.0 - e)
u_UO2 = 0.27003
M_U = M_UO2 * u_U / u_UO2

E_f = 3.708e-22
Nf = BU * M_U / E_f
N_av = 6.022e23
n_Xe_Kr = Nf * Y * R_r / N_av

R = 8.314
n_tot = n_Xe_Kr + n_N2 + n_H2O

V_min = n_tot * R * (T_ci[-1] + 273.15) / (pi_max * 1e6)
H_min = V_min / (np.pi * r_ci**2)
H_plenum = H_min + 0.15

p_H2O_id = n_H2O * R * (T_ci[-1] + 273.15) / V_min
p_sat_H2O = steamTable.psat_t(T_ci[-1]) * 1e5

if p_H2O_id < p_sat_H2O:
    print("\nIdeal-gas assumption for H2O is VALID: p_H2O_id < p_sat.")
else:
    print("\nWARNING: H2O would condense; ideal-gas assumption is NOT valid.")

print(f"p_H2O_id = {p_H2O_id:.4f} Pa | p_sat_H2O at T = {T_ci[-1]:.2f} °C: {p_sat_H2O:.4f} Pa")
print("\n--- GAS PLENUM SIZING ---")
print(f"Moles of Xe+Kr: {n_Xe_Kr:.6f} mol | N2: {n_N2:.6f} mol | H2O: {n_H2O:.6f} mol | Total: {n_tot:.6f} mol")
print(f"V_min: {V_min:.6e} m^3 | H_min: {H_min:.4f} m | H_plenum: {H_plenum:.4f} m")