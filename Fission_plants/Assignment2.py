import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.io import savemat
from pyXSteam.XSteam import XSteam

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

# %% Data and Parameters
heat_out_core = 3400.0           # [MWth]
f_fuel = 0.974                   # [-]
F_q = 2.60                       # [-]

H = 4.267                        # [m]
D_core = 3.040                   # [m]
D_pellet = 0.3225 * 2.54e-2      # [m]
D_co = 9.5e-3                    # [m]
rod_pitch = 12.6e-3              # [m]
N_fuel_rods = 41448              # [-]
A_ht = 41.8 * 0.0929             # [m^2]

Lr = 2.85e-2                     # [m]
lambda_tr = 0.0029               # [m]
Dc = 0.0966e-2                   # [m]
Dr = 0.16e-2                     # [m]

p_nom = 2250.0 * 0.0689476       # [bar]
p_min = 2190.0 * 0.0689476       # [bar]
T_inlet = 279.444                # [°C]
m_dot_vessel = 14300.0           # [Kg/s]
f_bp = 0.059                     # [-]
v_rod = 15.8 * 0.3048            # [m/s]

min_dnbr_typical = 2.80          # [-]
min_dnbr_thimble = 2.74          # [-]
alfa_hs_avg = 0.1                # [-]
alfa_hs_max = 0.9                # [-]

# %% Geometric and area calculations
Af = (np.pi / 4.0) * D_pellet**2
A_core = rod_pitch**2 - (np.pi / 4.0) * D_co**2
A_fuel = Af

A_fl = N_fuel_rods * A_core
D_eq = 4.0 * A_core / (np.pi * D_co)

delta = (Dc / Dr) * Lr
He = H + 1.42 * lambda_tr + 2.0 * delta

z = np.linspace(-H / 2.0, H / 2.0, 500)

# %% Point 1
qv_avg = heat_out_core * f_fuel / (Af * H * N_fuel_rods)  # [MW/m^3]
qv_max = qv_avg * F_q                                     # [MW/m^3]
qv = qv_max * np.cos(np.pi * z / He) * 1e6                # [W/m^3]

# %% Point 2
m_dot_core = m_dot_vessel * (1.0 - f_bp)
W_HC = m_dot_core / N_fuel_rods
G = m_dot_core / A_fl

# %% Point 3
i_in = steamTable.h_pt(p_nom, T_inlet)
i_ls = steamTable.hL_p(p_nom)
i_vs = steamTable.hV_p(p_nom)
T_sat = steamTable.tsat_p(p_nom)

i = i_in + 1.0267 * qv_max * 1e3 * Af * He / (W_HC * np.pi) * (np.sin(np.pi * z / He) + np.sin(np.pi * H / 2.0 / He))
i_avg = i_in + 1.0267 * qv_avg * 1e3 * Af * He / (W_HC * np.pi) * (np.sin(np.pi * z / He) + np.sin(np.pi * H / 2.0 / He))

# %% Point 4
T_cool = np.zeros_like(i)
T_cool_avg = np.zeros_like(i_avg)

for j in range(len(i)):
    if i[j] < i_ls:
        T_cool[j] = steamTable.t_ph(p_nom, i[j])
    else:
        T_cool[j] = T_sat
    T_cool_avg[j] = steamTable.t_ph(p_nom, i_avg[j])

plt.figure(1)
plt.plot(T_cool, z, color=[0.68, 0.78, 0.81], linewidth=2)
plt.grid(True)
plt.title('Coolant Temperature Profile - Hot Channel')
plt.xlabel('Temperature [°C]')
plt.ylabel('Axial Coordinate [m]')

plt.figure(2)
plt.plot(T_cool_avg, z, color=[0.86, 0.82, 1.0], linewidth=2)
plt.grid(True)
plt.xlim([275, 350])
plt.title('Coolant Temperature Profile - Average Channel')
plt.xlabel('Temperature [°C]')
plt.ylabel('Axial Coordinate [m]')

# %% Point 5
xi_eq = (i - i_ls) / (i_vs - i_ls)

plt.figure(3)
plt.plot(xi_eq, z, color=[1.0, 0.87, 0.73], linewidth=2)
plt.grid(True)
plt.title('Equilibrium Vapor Quality Profile')
plt.xlabel('Quality [-]')
plt.ylabel('Axial Coordinate [m]')

# %% Point 6
C = 0.042 * rod_pitch / D_co - 0.024
rho_l_sat = steamTable.rhoL_p(p_nom)
rho_v_sat = steamTable.rhoV_p(p_nom)
sigma = steamTable.st_t(T_sat)

mu = np.zeros_like(T_cool)
k = np.zeros_like(T_cool)
cp = np.zeros_like(T_cool)
Nu = np.zeros_like(T_cool)
h = np.zeros_like(T_cool)

for m in range(len(T_cool)):
    # In MATLAB m<=429 (1-based) equivale a m<=428 in Python (0-based)
    if m <= 428:
        mu[m] = steamTable.my_pt(p_nom, T_cool[m])
        k[m] = steamTable.tc_pt(p_nom, T_cool[m])
        cp[m] = steamTable.Cp_pt(p_nom, T_cool[m]) * 1e3
    else:
        mu[m] = mu[428]
        k[m] = k[428]
        cp[m] = cp[428]
    Nu[m] = C * (G * D_eq / mu[m])**0.8 * (cp[m] * mu[m] / k[m])**0.4
    h[m] = Nu[m] * k[m] / D_eq

q_surf = qv * (D_pellet**2) / (4.0 * D_co)

T_co_SP = T_cool + q_surf / h
T_co_JL = T_sat + 25.0 * (q_surf * 1e-6)**0.25 * np.exp(-p_nom / 62.0)

T_co = np.zeros_like(T_co_SP)
z_vec = []

for g_idx in range(len(T_co_SP)):
    if T_co_SP[g_idx] > T_co_JL[g_idx]:
        z_vec.append(g_idx)
        T_co[g_idx] = T_co_JL[g_idx]
    else:
        T_co[g_idx] = T_co_SP[g_idx]

savemat('Tco_data.mat', {'T_co': T_co})

z_sbc_idx = z_vec[0]
z_sbc = z[z_sbc_idx]

plt.figure(4)
plt.plot(T_co_SP, z, color=[0.8, 0.8, 0.8], linewidth=2, label='Single Phase (SP)')
plt.plot(T_co_JL, z, color=[1.0, 0.7, 0.72], linewidth=2, label='Jens-Lottes (JL)')
plt.plot(T_co, z, '--', color=[0.75, 0.99, 0.78], linewidth=2.5, label='Actual Cladding Temperature')
plt.grid(True)
plt.legend(loc='best')
plt.title('Outer Cladding Temperature Profile')
plt.xlabel('Temperature [°C]')
plt.ylabel('Axial Coordinate [m]')

# Liquid temperature at the detachment (Griffith model)
T_l_d = T_sat - q_surf / 5.0 / h

index_det = 0
for g_idx in range(len(T_cool)):
    if T_cool[g_idx] > T_l_d[g_idx]:
        index_det = g_idx
        break

z_det = z[index_det]

# Flow quality
T_l_d_mean = np.mean(T_l_d)
rho_l = steamTable.rhoL_t(T_l_d_mean)
rho_v = steamTable.rhoV_t(T_sat)

H_fg = (i_vs - i_ls) * 1e3
i_l = steamTable.h_pt(p_nom, T_l_d_mean)
eps = (rho_l / (rho_v_sat * H_fg)) * ((i_ls - i_l) * 1e3)
q_sp = h * (T_sat - T_l_d)
q_e = (q_surf - q_sp) / (1.0 + eps)

W = W_HC
coeff_int = H_fg * W * (1.0 + eps)
x = np.zeros_like(z)

for jj in range(len(z)):
    if z[jj] > z_det:
        integrand_func = lambda zz: np.pi * D_co * qv_max * 1e6 * np.cos(np.pi * zz / He) * (D_pellet**2 / 4.0) / D_co / coeff_int
        val, _ = quad(integrand_func, z_det, z[jj])
        x[jj] = val

# Void fraction
Rd = 2.37e-3 / (p_nom**0.237)
delta_vd = 0.0666 * Rd
Dh = 4.0 * A_core / np.pi / D_co
alfa_det = 4.0 * delta_vd / Dh

sigma = steamTable.st_p(p_nom)
C_0 = 1.13
C_1 = 1.4
g_acc = 9.81

coeff_zf = C_0 * (x / rho_v + (1.0 - x) / rho_l) + C_1 * (1.0 / G) * (sigma * g_acc * (rho_l - rho_v) / (rho_l**2))**0.25
alfa_z = (x / rho_v) / coeff_zf

# Slip ratio (nota: riproduce fedelmente la precedenza operatori MATLAB: (rho_l/rho_v)^1/2 = (rho_l/rho_v)/2)
S_f = (rho_l / rho_v)**1 / 2.0
alfa_s_f = x / (x + (1.0 - x) * (rho_v / rho_l) * S_f)

S_m = (rho_l / rho_v)**1 / 3.0
alfa_s_m = x / (x + (1.0 - x) * (rho_v / rho_l) * S_m)

alfa_z_new = np.zeros_like(z)
alfa_s_m_new = np.zeros_like(z)
alfa_s_f_new = np.zeros_like(z)

for ww in range(len(z)):
    if z[ww] <= z_sbc:
        alfa_z_new[ww] = 0.0
        alfa_s_m_new[ww] = 0.0
        alfa_s_f_new[ww] = 0.0
    elif z_sbc < z[ww] <= z_det:
        alfa_lin = alfa_det * (z[ww] - z_sbc) / (z_det - z_sbc)
        alfa_z_new[ww] = alfa_lin
        alfa_s_m_new[ww] = alfa_lin
        alfa_s_f_new[ww] = alfa_lin
    else:
        alfa_z_new[ww] = alfa_z[ww] + alfa_det
        alfa_s_m_new[ww] = alfa_s_m[ww] + alfa_det
        alfa_s_f_new[ww] = alfa_s_f[ww] + alfa_det

plt.figure(5)
plt.plot(z, alfa_z_new, '-', linewidth=2, label=r'$\alpha_{ZUBER}$')
plt.plot(z, alfa_s_f_new, '-', linewidth=2, label=r'$\alpha_{FAUSKE}$')
plt.plot(z, alfa_s_m_new, '-', linewidth=2, label=r'$\alpha_{MOODY}$')
plt.grid(True)
plt.title('Void fraction variation along the axial coordinate')
plt.xlabel('Axial coordinate [m]')
plt.ylabel(r'Void Fraction $\alpha$ [-]')
plt.legend(loc='upper left')

# %% Point 7 Inner cladding temperature
t_clad = 0.572e-3
D_ci = D_co - 2.0 * t_clad
k_cl = 11.45 + 1.425e-2 * T_co
T_ci = T_co + qv * A_fuel / (2.0 * np.pi * k_cl) * np.log(D_co / D_ci)

savemat('Tci_data.mat', {'T_ci': T_ci})

plt.figure(6)
plt.plot(T_ci, z, linewidth=2)
plt.grid(True)
plt.ylabel('z [m]')
plt.xlabel('T_ci [°C]')
plt.title('Inner cladding temperature')

# %% Point 8 Fuel pellet surface temperature
denominator_hg = 2.54e-5
sigma_rad = 5.670374e-8
eps_cl = 1.0
eps_f = 1.0
T_amb = 25.0
nu = 0.43
gamma = (D_co / 2.0) / (D_ci / 2.0)
delta_old = (D_ci / 2.0) - (D_pellet / 2.0)

T_fuel_old = T_ci + 40.0
toll = 1e-1
err = 10.0 * toll
p_in = 70e5  # Pa

while err > toll:
    h_rad = sigma_rad * ((T_fuel_old + 273.15)**4 - (T_ci + 273.15)**4) / (T_fuel_old - T_ci) * 1.0
    T_clad_ave = (np.mean(T_co) + np.mean(T_ci)) / 2.0
    T_fuel_ave = np.mean(T_fuel_old)

    alfa_f = 7.87e-6 + 3.9e-9 * T_fuel_old
    delta_f = (D_pellet * alfa_f * (T_fuel_ave - T_amb)) / 2.0

    alfa_cl = 5.62e-6 + 3.162e-9 * T_ci
    delta_cl = (D_ci * alfa_cl * (T_clad_ave - T_amb)) / 2.0

    E = 1.148e11 - 5.99e7 * (T_ci + 273.15)
    delta_cl_rad = (D_ci / 2.0) * (1.0 / E) * (1.0 / (gamma**2 - 1.0)) * (
        p_in * ((1.0 - nu) + (1.0 + nu) * gamma**2) - 2.0 * gamma**2 * (p_nom * 1e5)
    )

    delta_final = delta_old - delta_f - delta_cl - delta_cl_rad
    k_gas = 0.1763e-2 * ((T_fuel_old + 273.15 + T_ci + 273.15) / 2.0)**0.77163
    h_gap = k_gas / (denominator_hg + delta_final)

    T_fs_rad = T_ci + qv * A_fuel / (np.pi * D_pellet * (h_gap + h_rad))
    T_fs_new = T_ci + qv * A_fuel / (np.pi * D_pellet * h_gap)

    err = np.max(np.abs(T_fs_new - T_fuel_old) / np.abs(T_fs_new))
    T_fuel_old = T_fs_rad.copy()

savemat('p_in_data.mat', {'p_in': p_in})

plt.figure(11)
plt.plot(T_fs_rad, z, linewidth=2, label=r'$T_{fuel,rad}$')
plt.plot(T_fs_new, z, linewidth=2, label=r'$T_{fuel}$')
plt.grid(True)
plt.title('Variation in space of the fuel pellet surface temperature')
plt.ylabel('z [m]')
plt.xlabel('Temperature [°C]')
plt.legend(loc='upper left')

if np.all(delta_final >= 0):
    print("There is no contact between fuel and cladding inner surface")
else:
    print("There is a contact between fuel and cladding inner surface")

# %% Point 9 Fuel centerline temperature
f = 0.97
T_fcl = np.zeros_like(z)

for ii in range(len(z)):
    qv_new = qv[ii]
    coef = (qv_new * A_fuel) / (4.0 * np.pi) * f
    residual = lambda T: 100.0 * (
        (1.0 / 0.0238) * np.log((11.8 + 0.0238 * T) / (11.8 + 0.0238 * T_fs_rad[ii]))
        + 2.19375e-13 * (T**4 - T_fs_rad[ii]**4)
    ) - coef
    T_guess = T_fs_rad[ii] + 300.0
    T_fcl[ii] = fsolve(residual, T_guess, xtol=1e-4)[0]

plt.figure(12)
plt.plot(T_fcl, z, linewidth=2)
plt.grid(True)
plt.ylabel('z [m]')
plt.xlabel('T_fcl [°C]')
plt.title('Fuel centerline temperature')

plt.figure(13)
plt.plot(T_cool, z, linewidth=2, label=r'$T_{coolant}$')
plt.plot(T_co_SP, z, linewidth=2, label=r'$T_{cl,outer}$')
plt.plot(T_ci, z, linewidth=2, label=r'$T_{cl,inner}$')
plt.plot(T_fs_rad, z, linewidth=2, label=r'$T_{fp,surface}$')
plt.plot(T_fcl, z, linewidth=2, label=r'$T_{fp,centerline}$')
plt.grid(True)
plt.title('Axial temperature profiles')
plt.xlabel('Temperature [°C]')
plt.ylabel('Axial coordinate [m]')
plt.legend()

# %% Point 10 Evaluation of the critical heat flux
qc = (
    ((2.022 - 0.06238 * p_nom * 1e-1) + (0.1722 - 0.01427 * p_nom * 1e-1) * np.exp((18.177 - 0.5987 * p_nom * 1e-1) * xi_eq))
    * ((0.1484 - 1.596 * xi_eq + 0.1729 * xi_eq * np.abs(xi_eq)) * 2.326 * G + 3271.0)
    * (1.157 - 0.869 * xi_eq)
    * (0.2664 + 0.8357 * np.exp(-124.1 * D_eq))
    * (0.8285 + 0.0003413 * (i_ls - i_in))
)

Ls = H / 8.0 * 39.3701
ks = 0.066 + (21.0 - 20.0) * (0.046 - 0.066) / (26.0 - 20.0)
Fs = (2250.0008834 / 225.896)**0.5 * (1.445 - 0.0371 * 13.999344) * (np.exp((xi_eq + 0.2)**2) - 0.73) + ks * 2550000.0 / 10e6 * (0.038 / 0.019)**0.35
correction_factor = 0.88
qc_gf = qc * Fs * correction_factor * 1e3

F = np.zeros_like(z)
q_c_nu = np.zeros_like(z)
conversion = 0.0254

for idx in range(z_sbc_idx + 1, len(z)):
    Lc_nu = z[idx] - z_sbc
    x_local = xi_eq[idx]

    G_new = 2.55e6
    C_in = 0.15 * ((1.0 - x_local)**4.31) / ((G_new / 1e6)**0.478)
    C_m = C_in / conversion

    z_slice = z[z_sbc_idx : idx + 1]
    denomin = np.cos(np.pi * z[idx] / He) * (1.0 - np.exp(-C_m * Lc_nu))
    integrand = np.cos(np.pi * z_slice / He) * np.exp(-C_m * (z[idx] - z_slice))

    if len(z_slice) >= 2:
        F[idx] = C_m * np.trapezoid(integrand, z_slice) / denomin
        q_c_nu[idx] = qc_gf[idx] / F[idx]

# %% Point 11 Evaluation of the DNBR
DNBR = q_c_nu / q_surf
valid_dnbr = DNBR[DNBR > 0]
MDNBR = np.min(valid_dnbr)
idx_min = np.where(DNBR == MDNBR)[0][0]
z_MDNBR = z[idx_min]

print(f"\nThe minimum value of the DNBR is: {MDNBR:.4f} (at z = {z_MDNBR:.4f} m)")

q_c_nu_plot = q_c_nu.copy()
q_c_nu_plot[q_c_nu_plot == 0] = np.nan

plt.figure(14)
plt.plot(z, qc_gf, linewidth=2, label='UHF-W3 (Uniform Grid Factor)')
plt.plot(z, q_c_nu_plot, linewidth=2, label='CHF-Non Uniform (Tong)')
plt.plot(z, q_surf, linewidth=2, label=r'$q_{surf}$')
plt.grid(True)
plt.title('Distribuzione Assiale delle Critical Heat Fluxes')
plt.xlabel('Coordinata Assiale z [m]')
plt.ylabel('Heat Flux [W/m^2]')
plt.legend(loc='best')

plt.show()