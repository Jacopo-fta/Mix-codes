import numpy as np
from pyXSteam.XSteam import XSteam

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

# %% PSC DATA
p_PSC = 75.0                     # [bar] Coolant pressure
D_out_PSC_in = 16.0              # [inches] Outer diameter (Schedule 100)
L_PSC = 16.0                     # [m] Total pipe length
rel_rough_PSC = 2e-4             # [-] Relative roughness epsilon/D
k_bend_PSC = 0.45                # [-] Loss coeff. for 90° bends
N_bends_PSC = 4                  # [-] Number of 90° bends
k_valve_PSC = 0.12               # [-] Check valve loss coeff.
H1_PSC = 7.0                     # [m] Elevation between hot and cold leg
H2_core = 3.0                    # [m] Elevation between core top and bottom

D_out_PSC = D_out_PSC_in * 0.0254
t_pipe_PSC = 1.031 * 0.0254
D_in_pipe_PSC = D_out_PSC - 2.0 * t_pipe_PSC
A_pipe_PSC = (np.pi * D_in_pipe_PSC**2) / 4.0

DP_vessel_nom = 1.2e5            # [Pa] (1.2 bar)
m_dot_nom = 3200.0               # [kg/s]
rho_nom = 700.0                  # [kg/m^3]
K_vessel_eq = (DP_vessel_nom * rho_nom) / (m_dot_nom**2)

# %% HX1 DATA
D_out_tube_HX1 = 19.05 / 1000.0  # [m]
t_tube_HX1 = 1.24 / 1000.0       # [m]
N_t_HX1 = 897                    # [-]
d_p_HX1 = 28.5 / 1000.0          # [m]
D_shell_HX1 = 1.5                # [m]
N_b_HX1 = 2                      # [-]
l_baffles_HX1 = 1.6              # [m]
L_tubes_HX1 = 9.314              # [m]
k_th_tube_HX1 = 15.0             # [W/mK]
A_h_HX1 = 0.883                  # [m^2]
A_ht_HX1 = 500.0                 # [m^2]
rel_rough_tube_HX1 = 1e-4        # [-]
F_T_HX1 = 0.7                    # [-]

D_in_tube_HX1 = D_out_tube_HX1 - 2.0 * t_tube_HX1
A_tube_single = (np.pi * D_in_tube_HX1**2) / 4.0
A_tube_total = N_t_HX1 * A_tube_single

D_eq_shell = (2.0 * np.sqrt(3.0) * d_p_HX1**2) / (np.pi * D_out_tube_HX1) - D_out_tube_HX1
A_shell = (D_shell_HX1 / d_p_HX1) * (d_p_HX1 - D_out_tube_HX1) * l_baffles_HX1

k_exp1 = (1.0 - A_pipe_PSC / A_h_HX1)**2
k_contr1 = 0.5 * (1.0 - A_tube_total / A_h_HX1)
k_exp2 = (1.0 - A_tube_total / A_h_HX1)**2
k_contr2 = 0.5 * (1.0 - A_pipe_PSC / A_h_HX1)

# %% ISC DATA
p_ISC = 70.0                     # [bar]
D_out_ISC_in = 16.0              # [inches]
L_pipe_ISC = 40.0                # [m]
k_bend_ISC = 0.45                # [-]
N_bends_ISC = 2                  # [-]
rel_rough_ISC = 2e-4             # [-]
H_ISC = 10.0                     # [m]

t_pipe_ISC_in = 1.031
t_pipe_ISC = t_pipe_ISC_in * 0.0254
D_out_ISC = D_out_ISC_in * 0.0254
D_in_tube_ISC = D_out_ISC - 2.0 * t_pipe_ISC
A_pipe_ISC = (np.pi * D_in_tube_ISC**2) / 4.0

# %% HX2 DATA
D_out_tube_HX2 = 25.4 / 1000.0   # [m]
t_tube_HX2 = 1.24 / 1000.0       # [m]
N_t_HX2 = 770                    # [-]
L_tubes_HX2 = 7.0                # [m]
D_manifold_HX2_in = 16.0         # [inches]
k_th_tube_HX2 = 15.0             # [W/mK]
A_ht_HX2 = 430.0                 # [m^2]
rel_rough_tube_HX2 = 1e-4        # [-]

D_in_tube_HX2 = D_out_tube_HX2 - 2.0 * t_tube_HX2
A_tube_single_HX2 = (np.pi * D_in_tube_HX2**2) / 4.0
A_tube_total_HX2 = N_t_HX2 * A_tube_single_HX2

D_manifold_HX2 = D_manifold_HX2_in * 0.0254
A_manifold_HX2 = (np.pi * D_manifold_HX2**2) / 4.0

# %% Other data for the solution
g = 9.81
P_nom = 600e6
P_decay = 0.01 * P_nom           # 6 MW
T_sat_pool = 100.0

q_pp = P_decay / A_ht_HX2
DT_sat_HX2 = (q_pp / 2.257)**(1.0 / 3.86)
T_wall_HX2 = T_sat_pool + DT_sat_HX2
T_sat = T_wall_HX2 - DT_sat_HX2
h_out = q_pp / DT_sat_HX2

# %% Calculation of ISC temperatures for the first guess
m_dot_isc = 100.0
T_av_isc = 120.0

cp_guess1 = steamTable.Cp_pt(p_ISC, T_av_isc) * 1000.0
DT_guess1 = q_pp / (m_dot_isc * cp_guess1)
T_hot_isc = T_av_isc + (DT_guess1 / 2.0)
T_cold_isc = T_av_isc - (DT_guess1 / 2.0)

# %% ISC Loop
tol_isc = 1e-3
iter_isc = 0
diff_isc = 1.0

while diff_isc > tol_isc and iter_isc < 1000:
    iter_isc += 1

    rho_av_isc = steamTable.rho_pt(p_ISC, T_av_isc)
    mu_isc = steamTable.my_pt(p_ISC, T_av_isc)
    cp_isc_KJ = steamTable.Cp_pt(p_ISC, T_av_isc)
    cp_isc_J = cp_isc_KJ * 1000.0
    kth_fluid_isc = steamTable.tc_pt(p_ISC, T_av_isc)
    rho_hot_isc = steamTable.rho_pt(p_ISC, T_hot_isc)
    rho_cold_isc = steamTable.rho_pt(p_ISC, T_cold_isc)

    v_pipe_ISC = m_dot_isc / (rho_av_isc * A_pipe_ISC)
    v_HX2 = m_dot_isc / (rho_av_isc * A_tube_total_HX2)
    v_shell = m_dot_isc / (rho_av_isc * A_shell)

    Re_pipe = (rho_av_isc * v_pipe_ISC * D_in_tube_ISC) / mu_isc
    Re_HX2 = (rho_av_isc * v_HX2 * D_in_tube_HX2) / mu_isc
    Re_shell = (rho_av_isc * v_shell * D_eq_shell) / mu_isc

    Pr_isc = (mu_isc * cp_isc_J) / kth_fluid_isc
    Nu_HX2_isc = (0.023 * Re_HX2**0.8) * (Pr_isc**0.3)
    h_in = (Nu_HX2_isc * kth_fluid_isc) / D_in_tube_HX2

    R_conv_in_spec = 1.0 / h_in
    R_conv_out_spec = 1.0 / h_out
    R_cond_wall_hx1_spec = (D_out_tube_HX1 * np.log(D_out_tube_HX1 / D_in_tube_HX1)) / (2.0 * k_th_tube_HX2)

    U_hx2 = 1.0 / (R_cond_wall_hx1_spec + R_conv_out_spec + R_conv_in_spec)

    T_cold_isc = T_sat + (T_hot_isc - T_cold_isc) / (np.exp(U_hx2 * A_ht_HX2 / (m_dot_isc * cp_isc_J)) - 1.0)

    DT_isc = P_decay / (m_dot_isc * cp_isc_J)
    T_hot_isc = T_cold_isc + DT_isc
    T_av_isc = (T_hot_isc + T_cold_isc) / 2.0

    f_pipe = (-1.8 * np.log10((rel_rough_ISC / 3.7)**1.11 + 6.9 / Re_pipe))**(-2)
    f_HX2 = (-1.8 * np.log10((rel_rough_tube_HX2 / 3.7)**1.11 + 6.9 / Re_HX2))**(-2)

    DP_fric_pipe = f_pipe * (L_pipe_ISC / D_in_tube_ISC) * (rho_av_isc * v_pipe_ISC**2 / 2.0)
    DP_fric_HX2 = f_HX2 * (L_tubes_HX2 / D_in_tube_HX2) * (rho_av_isc * v_HX2**2 / 2.0)

    K_shell_HX1 = 0.227 / (Re_shell**0.193)
    DP_loc_shell = K_shell_HX1 * (rho_av_isc * v_shell**2 / 2.0)

    DP_loc_bends = N_bends_ISC * k_bend_ISC * (rho_av_isc * v_pipe_ISC**2 / 2.0)

    DP_total_loss = DP_fric_pipe + DP_fric_HX2 + DP_loc_shell + DP_loc_bends
    DP_buoyancy = (rho_cold_isc - rho_hot_isc) * g * H_ISC

    diff_isc = abs(DP_buoyancy - DP_total_loss)
    m_dot_isc = m_dot_isc * np.sqrt(DP_buoyancy / DP_total_loss)

    if iter_isc == 1:
        print(f"Portata prima iterazione ISC (m_dot_isc): {m_dot_isc:.2f} kg/s")

print(f"# Iterazioni ISC {iter_isc:.2f}")
print("\n=== Results ISC ===")
print(f"Final mass flow (m_dot_isc): {m_dot_isc:.2f} kg/s")
print(f"Temperature: T_hot = {T_hot_isc:.2f} °C | T_cold = {T_cold_isc:.2f} °C | T_av = {T_av_isc:.2f} °C")
print(f"DP_buoyancy: {DP_buoyancy:.2f} Pa")
print(f"DP_total_loss: {DP_total_loss:.2f} Pa")
print(f"Final error: {diff_isc:.4f} Pa")

Q_check = m_dot_isc * cp_isc_J * (T_hot_isc - T_cold_isc)
print(f"Disposed Power (must be ~6.00e+06 W): {Q_check:.2e} W")
print("==========================================\n")

# %% Ciclo PSC
DT_isc = T_hot_isc - T_cold_isc
Pr_isc_final = (mu_isc * cp_isc_J) / kth_fluid_isc
Nu_shell_HX1 = 0.351 * (Re_shell**0.55) * (Pr_isc_final**(1.0 / 3.0))
h_shell_HX1 = (Nu_shell_HX1 * kth_fluid_isc) / D_eq_shell

m_dot_psc = m_dot_isc
T_av_psc = 250.0
cp_guess_psc_j = steamTable.Cp_pt(p_PSC, T_av_psc) * 1000.0
DT_guess_psc = P_decay / (m_dot_psc * cp_guess_psc_j)

T_hot_psc = T_av_psc + (DT_guess_psc / 2.0)
T_cold_psc = T_av_psc - (DT_guess_psc / 2.0)

tol_psc = 1e-3
iter_psc = 0
diff_psc = 1000.0

while diff_psc > tol_psc and iter_psc < 1000:
    iter_psc += 1

    rho_av_psc = steamTable.rho_pt(p_PSC, T_av_psc)
    mu_psc = steamTable.my_pt(p_PSC, T_av_psc)
    cp_psc_J = steamTable.Cp_pt(p_PSC, T_av_psc) * 1000.0
    kth_fluid_psc = steamTable.tc_pt(p_PSC, T_av_psc)

    v_tube_HX1 = m_dot_psc / (rho_av_psc * A_tube_total)
    Re_tube_HX1 = (rho_av_psc * v_tube_HX1 * D_in_tube_HX1) / mu_psc
    Pr_psc = (mu_psc * cp_psc_J) / kth_fluid_psc

    Nu_in_HX1 = 0.023 * (Re_tube_HX1**0.8) * (Pr_psc**0.3)
    h_in_HX1 = (Nu_in_HX1 * kth_fluid_psc) / D_in_tube_HX1

    R_conv_in_HX1 = 1.0 / h_in_HX1
    R_conv_out_HX1 = 1.0 / h_shell_HX1
    R_cond_wall_HX1 = (D_out_tube_HX1 * np.log(D_out_tube_HX1 / D_in_tube_HX1)) / (2.0 * k_th_tube_HX1)

    U_hx1 = 1.0 / (R_cond_wall_HX1 + R_conv_out_HX1 + R_conv_in_HX1)

    DT_psc = P_decay / (m_dot_psc * cp_psc_J)
    T_cold_psc = T_cold_isc + (DT_psc - DT_isc) / (
        np.exp((U_hx1 * A_ht_HX1 * F_T_HX1 * (DT_psc - DT_isc)) / P_decay) - 1.0
    )

    T_hot_psc = T_cold_psc + DT_psc
    T_av_psc = (T_hot_psc + T_cold_psc) / 2.0

    rho_hot_psc = steamTable.rho_pt(p_PSC, T_hot_psc)
    rho_cold_psc = steamTable.rho_pt(p_PSC, T_cold_psc)

    integral_rho_core = H2_core * (rho_cold_psc + rho_hot_psc) / 2.0
    DP_buoyancy_psc = (rho_cold_psc * g * (H1_PSC + H2_core)) - (integral_rho_core * g + rho_hot_psc * g * H1_PSC)

    v_pipe_psc = m_dot_psc / (rho_av_psc * A_pipe_PSC)
    Re_pipe_psc = (rho_av_psc * v_pipe_psc * D_in_pipe_PSC) / mu_psc

    f_pipe_psc = (-1.8 * np.log10((rel_rough_PSC / 3.7)**1.11 + 6.9 / Re_pipe_psc))**(-2)
    f_HX1 = (-1.8 * np.log10((rel_rough_tube_HX1 / 3.7)**1.11 + 6.9 / Re_tube_HX1))**(-2)

    DP_fric_pipe_psc = f_pipe_psc * (L_PSC / D_in_pipe_PSC) * (rho_av_psc * v_pipe_psc**2 / 2.0)
    DP_fric_HX1 = f_HX1 * (L_tubes_HX1 / D_in_tube_HX1) * (rho_av_psc * v_tube_HX1**2 / 2.0)
    DP_loc_psc = (N_bends_PSC * k_bend_PSC + k_valve_PSC) * (rho_av_psc * v_pipe_psc**2 / 2.0)

    DP_total_loss_psc = DP_fric_pipe_psc + DP_fric_HX1 + DP_loc_psc
    diff_psc = abs(DP_buoyancy_psc - DP_total_loss_psc)

    m_dot_psc = m_dot_psc * np.sqrt(DP_buoyancy_psc / DP_total_loss_psc)

    if iter_psc == 1:
        print(f"Portata prima iterazione (m_dot_psc): {m_dot_psc:.2f} kg/s")

print(f"# Iterazione PSC {iter_psc:.2f}")
print("\n=== Results PSC ===")
print(f"Final mass flow (m_dot_psc): {m_dot_psc:.2f} kg/s")
print(f"Temperature: T_hot = {T_hot_psc:.2f} °C | T_cold = {T_cold_psc:.2f} °C | T_av = {T_av_psc:.2f} °C")
print(f"DP_buoyancy: {DP_buoyancy_psc:.2f} Pa")
print(f"DP_total_loss: {DP_total_loss_psc:.2f} Pa")

Q_check_psc = m_dot_psc * cp_psc_J * (T_hot_psc - T_cold_psc)
print(f"Disposed Power (must be ~6.00e+06 W): {Q_check_psc:.2e} W")
print("==========================================\n")