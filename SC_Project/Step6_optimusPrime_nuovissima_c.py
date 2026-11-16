# =========================================================================
# Step6_optimusPrime_nuovissima_c.py
# =========================================================================
# Master script integrato:
#   PHASE 0  : diagnostica baseline a T = 20 K (sizing, EM, perdite AC)
#   PHASE 1  : bisezione mdot per portata minima con margine >= 5 K
#   PHASE 2  : transient finale 2-way coupled EM<->TH a mdot_min
#
# Caratteristiche:
#   - Step 2 (Biot-Savart + L_dp via inductanceMatrix1d) caricato da MagneticData_Final_nuovissima.mat
#   - Step 4 (Brandt hysteresis + Cu eddy + Ohmic SC power-law + Coupling) interno
#   - Step 5 (3-eq TH, BDF1 implicito + upwind He) accoppiato a 2 vie
#   - Circuito risolto a OGNI dt con directSolve (R(T), Jc(B,T))
#   - Quench check: termina early se T_S2 > T_cs + 5 K
# =========================================================================
import os
import math
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import PchipInterpolator, interp1d
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

from setupProject_nuovissima import setupProject_nuovissima, ProjectStruct
from cprintf import cprintf
from biot_savart_loops_vec import biot_savart_loops_vec
from directSolve import directSolve
from dimensionamento_CICC_nuovissima import run_dimensionamento_CICC
from Step2_MagneticDesign_nuovissima import run_step2_magnetic_design, _load_cicc_geometry


# =========================================================================
#                         THERMOPHYSICAL PROPERTIES
# =========================================================================

def rho_Cu_local(T, RRR):
    """
    RHO_CU_LOCAL  Resistivita' elettrica del rame OFHC (Matthiessen + NIST Cryogenic Fit).
    Valido per T in [4, 300] K e per qualsiasi valore di RRR.
    """
    T = np.clip(np.asarray(T, dtype=float), 4.0, 300.0)
    rho_residual = 1.68e-8 / RRR  # [Ohm*m]

    P1 = 1.171e-17
    P2 = 4.49
    P3 = 3.841e-10
    P4 = 1.14

    rho_phonon = (P1 * (T ** P2)) / (1.0 + P3 * (T ** P4))
    return rho_residual + rho_phonon


def props_SS(T):
    """
    PROPS_SS  Proprieta' termofisiche Acciaio Inox AISI 316LN (NIST Cryogenic Database).
    Valido per T in [4, 300] K.
    """
    T = np.clip(np.asarray(T, dtype=float), 4.0, 300.0)
    rho = 7950.0 * np.ones_like(T)
    logT = np.log10(T)

    # --- 1. Calore Specifico cp [J/(kg*K)] ---
    a_c = 22.0061
    b_c = -127.5528
    c_c = 303.2428
    d_c = -382.4167
    e_c = 277.5363
    f_c = -118.5794
    g_c = 27.8931
    h_c = -2.8028

    log10_cp = (
        a_c
        + b_c * logT
        + c_c * (logT ** 2)
        + d_c * (logT ** 3)
        + e_c * (logT ** 4)
        + f_c * (logT ** 5)
        + g_c * (logT ** 6)
        + h_c * (logT ** 7)
    )
    cp = 10.0 ** log10_cp

    # --- 2. Conducibilita' Termica k [W/(m*K)] ---
    a_k = -1.4087
    b_k = 1.3982
    c_k = 0.2543
    d_k = -0.6260
    e_k = 0.2334
    f_k = 0.4256
    g_k = -0.4658
    h_k = 0.1650
    i_k = -0.0199

    log10_k = (
        a_k
        + b_k * logT
        + c_k * (logT ** 2)
        + d_k * (logT ** 3)
        + e_k * (logT ** 4)
        + f_k * (logT ** 5)
        + g_k * (logT ** 6)
        + h_k * (logT ** 7)
        + i_k * (logT ** 8)
    )
    k = 10.0 ** log10_k
    return rho, cp, k


def props_Cu(T):
    """
    PROPS_CU  Proprieta' termofisiche rame OFHC (NIST Cryogenic Database).
    Valido per T in [4, 300] K e RRR = 100.
    """
    T = np.clip(np.asarray(T, dtype=float), 4.0, 300.0)
    rho = 8960.0 * np.ones_like(T)

    # --- 1. Calore Specifico cp [J/(kg*K)] - Polinomio di log10(T) ---
    a_cp = -1.91844
    b_cp = -0.15973
    c_cp = 8.61013
    d_cp = -18.996
    e_cp = 21.9661
    f_cp = -12.7328
    g_cp = 3.54322
    h_cp = -0.3797

    logT = np.log10(T)
    log10_cp = (
        a_cp
        + b_cp * logT
        + c_cp * (logT ** 2)
        + d_cp * (logT ** 3)
        + e_cp * (logT ** 4)
        + f_cp * (logT ** 5)
        + g_cp * (logT ** 6)
        + h_cp * (logT ** 7)
    )
    cp = 10.0 ** log10_cp

    # --- 2. Conducibilita' Termica k [W/(m*K)] per RRR = 100 ---
    a_k = 2.2154
    b_k = -0.47461
    c_k = -0.88068
    d_k = 0.13871
    e_k = 0.29505
    f_k = -0.02043
    g_k = -0.04831
    h_k = 0.001281
    i_k = 0.003207

    T05 = np.sqrt(T)
    T15 = T * T05
    T20 = T ** 2
    num_k = a_k + c_k * T05 + e_k * T + g_k * T15 + i_k * T20
    den_k = 1.0 + b_k * T05 + d_k * T + f_k * T15 + h_k * T20
    log10_k = num_k / den_k
    k = 10.0 ** log10_k
    return rho, cp, k


def props_S2(T, frac):
    """
    PROPS_S2  Proprietà termofisiche equivalenti Cu + SC (Eq. 2.26 e 2.27).
    Media pesata sulle frazioni d'area tra Rame (former + stabilizzatore) e REBCO.
    """
    rCu, cCu, kCu = props_Cu(T)

    rSC = 6200.0 * np.ones_like(rCu)
    cSC = 0.5 * cCu
    kSC = 5.0 * np.ones_like(kCu)

    fCu = frac.Cu
    fSC = frac.SC

    k = fCu * kCu + fSC * kSC
    rho = fCu * rCu + fSC * rSC

    rhocp = fCu * (rCu * cCu) + fSC * (rSC * cSC)
    cp = rhocp / rho
    return rho, cp, k


_HE_LOOKUP = {}


def props_He(T):
    """
    PROPS_HE  Proprieta' termofisiche dell'elio supercritico a p = 5 bar via CoolProp.
    Genera una tabella continua (PchipInterpolator su lookup table generata da CoolProp,
    identica a interp1(..., 'pchip') di MATLAB).
    """
    if not _HE_LOOKUP:
        p_He = 5e5  # [Pa] = 5 bar
        T_grid = np.linspace(4.5, 300.0, 300)
        rho_grid = np.array([CP.PropsSI("Dmass", "T", float(t), "P", p_He, "Helium") for t in T_grid])
        cp_grid = np.array([CP.PropsSI("Cpmass", "T", float(t), "P", p_He, "Helium") for t in T_grid])
        k_grid = np.array([CP.PropsSI("conductivity", "T", float(t), "P", p_He, "Helium") for t in T_grid])
        mu_grid = np.array([CP.PropsSI("viscosity", "T", float(t), "P", p_He, "Helium") for t in T_grid])

        _HE_LOOKUP["rho"] = PchipInterpolator(T_grid, rho_grid)
        _HE_LOOKUP["cp"] = PchipInterpolator(T_grid, cp_grid)
        _HE_LOOKUP["k"] = PchipInterpolator(T_grid, k_grid)
        _HE_LOOKUP["mu"] = PchipInterpolator(T_grid, mu_grid)

    T_clamped = np.clip(np.asarray(T, dtype=float), 4.5, 300.0)
    rho = _HE_LOOKUP["rho"](T_clamped)
    cp = _HE_LOOKUP["cp"](T_clamped)
    k = _HE_LOOKUP["k"](T_clamped)
    mu = _HE_LOOKUP["mu"](T_clamped)
    return rho, cp, k, mu


def q2str(flag):
    return " [QUENCH]" if flag else ""


# =========================================================================
#                         LOSSES, CIRCUIT & 3-EQ SOLVER
# =========================================================================

def calc_losses_full(B_eff_now, dBperp, dBmod, T_S2, R_DP_x, ACk):
    """
    CALC_LOSSES_FULL  Sorgenti di calore istantanee.
        q_hyst : isteresi REBCO (Brandt thin strip)
        q_eddy : eddy currents nel Cu former (full transversal)
        q_ohm  : Joule SC in regime power-law
        q_coup : coupling losses inter-tape (time-constant model, Wilson)
    """
    F_T = ((ACk.Tc - T_S2) / ACk.Tc) ** ACk.s_scaling
    F_T = np.maximum(F_T, 0.0)

    J_c0T = ACk.J_c0_base * F_T
    J_c = J_c0T / ((1.0 + B_eff_now / ACk.B0_op) ** ACk.alpha_op)
    J_c = np.maximum(J_c, 1e-3)

    # --- 2) Eddy nel Cu former (T-dipendente via rho_Cu) -----------------
    rho_Cu_T = rho_Cu_local(T_S2, ACk.RRR_cu)
    P_eddy_vol_Cu = (ACk.R_former ** 2) * (dBmod ** 2) / (8.0 * rho_Cu_T)
    q_eddy = (P_eddy_vol_Cu * ACk.A_Cu_net) / ACk.A_S2

    # --- 1) Hysteresis (Brandt thin strip, istantanea) -------------------
    P_hyst_vol_REBCO = (2.0 / np.pi) * J_c * ACk.w_hyst_eff * np.abs(dBperp)
    P_hyst_lin = P_hyst_vol_REBCO * ACk.A_REBCO_film
    q_hyst = P_hyst_lin / ACk.A_S2

    # --- 3) Ohmic via power-law (J_op / J_c)^(n-1) -----------------------
    J_op = ACk.I_op / ACk.A_REBCO_geom
    with np.errstate(over="ignore"):
        rho_pl = (ACk.E_c / J_c) * ((J_op / J_c) ** (ACk.n_pl - 1))
    rho_pl_max = rho_Cu_T * (ACk.A_REBCO_geom / ACk.A_Cu_total)
    rho_pl = np.minimum(rho_pl, rho_pl_max)
    q_ohm_REBCO = rho_pl * (J_op ** 2)
    q_ohm = q_ohm_REBCO * ACk.A_REBCO_film / ACk.A_S2

    # --- 4) Coupling losses inter-tape (Wilson time-constant, cavo TWISTATO) -----
    mu_0_const = 4.0 * np.pi * 1e-7
    q_coup_REBCO = (ACk.tau_coupling / mu_0_const) * (dBperp ** 2)
    q_coup = q_coup_REBCO * ACk.A_REBCO_film / ACk.A_S2

    # --- Totale --------------------------------------------------------
    q_AC = q_hyst + q_eddy + q_ohm + q_coup

    # Resistance per nodo (per il circuit solver downstream)
    L_per_turn_x = 2.0 * np.pi * R_DP_x
    R_per_x = rho_pl * L_per_turn_x / ACk.A_REBCO_geom
    return q_AC, q_hyst, q_eddy, q_ohm, q_coup, R_per_x


_CIRCUIT_CACHE = {}


def solve_circuit_directSolve(R_per_turn, L_dp, M_ext_per_turn, I_op, dIdt):
    """
    SOLVE_CIRCUIT_DIRECTSOLVE  Risolve la rete circuitale del DP via directSolve.
    Sistema square N_nodi x N_nodi (sparse) con vincolo Dirichlet phi(OUT)=0.
    """
    N_sp_dp = R_per_turn.size
    N_nodi = N_sp_dp + 1

    DV_edge = R_per_turn * I_op + np.sum(L_dp, axis=1) * dIdt + M_ext_per_turn * dIdt

    if N_sp_dp not in _CIRCUIT_CACHE:
        rows = np.concatenate([np.arange(N_sp_dp), np.arange(1, N_sp_dp + 1)])
        cols = np.concatenate([np.arange(N_sp_dp), np.arange(N_sp_dp)])
        vals = np.concatenate([np.ones(N_sp_dp), -np.ones(N_sp_dp)])
        A_inc = sp.csr_matrix((vals, (rows, cols)), shape=(N_nodi, N_sp_dp))

        A_full = sp.lil_matrix((N_nodi, N_nodi), dtype=float)
        A_full[:N_sp_dp, :] = A_inc.T
        A_full[N_nodi - 1, N_nodi - 1] = 1.0
        A_full_csr = A_full.tocsr()

        b_dummy = np.zeros(N_nodi, dtype=float)
        _, factor_cached = directSolve(A_full_csr, b_dummy, N_nodi - 1, 0.0, "verbose", "n")
        _CIRCUIT_CACHE[N_sp_dp] = (A_full_csr, factor_cached)

    A_full_csr, factor_cached = _CIRCUIT_CACHE[N_sp_dp]
    b_full = np.zeros(N_nodi, dtype=float)
    b_full[:N_sp_dp] = DV_edge

    phi, _ = directSolve(
        A_full_csr, b_full, N_nodi - 1, 0.0, "factor", factor_cached, "verbose", "n"
    )
    return phi


def build_3eq_matrix(T_J, T_S2, T_He, aJ, aS2, bJ, bS2J, bS2He, bHeS2, C, gS2, T_in, N):
    """
    Costruzione vettorizzata della matrice 3N x 3N del sistema termoidraulico 3-equazioni
    (identica elemento per elemento a build_3eq_matrix di MATLAB).
    """
    idx = np.arange(N, dtype=int)
    iJ = idx
    iS2 = N + idx
    iHe = 2 * N + idx

    # --- 1) Equazione Jacket (J) ---
    d_J = 1.0 + 2.0 * aJ + bJ
    cM_J = np.where(idx == N - 1, -2.0 * aJ, -aJ)
    cP_J = np.where(idx == 0, -2.0 * aJ, -aJ)

    rows_J = [iJ, iJ[1:], iJ[:-1], iJ]
    cols_J = [iJ, iJ[:-1], iJ[1:], iS2]
    vals_J = [d_J, cM_J[1:], cP_J[:-1], -bJ]

    # --- 2) Equazione Conduttore (S2) ---
    d_S2 = 1.0 + 2.0 * aS2 + bS2J + bS2He
    cM_S2 = np.where(idx == N - 1, -2.0 * aS2, -aS2)
    cP_S2 = np.where(idx == 0, -2.0 * aS2, -aS2)

    rows_S2 = [iS2, iS2[1:], iS2[:-1], iS2, iS2]
    cols_S2 = [iS2, iS2[:-1], iS2[1:], iJ, iHe]
    vals_S2 = [d_S2, cM_S2[1:], cP_S2[:-1], -bS2J, -bS2He]

    # --- 3) Equazione Elio (He) ---
    d_He = np.empty(N, dtype=float)
    d_He[0] = 1.0
    d_He[1:] = 1.0 + C[1:] + bHeS2[1:]

    rows_He = [iHe, iHe[1:], iHe[1:]]
    cols_He = [iHe, iHe[:-1], iS2[1:]]
    vals_He = [d_He, -C[1:], -bHeS2[1:]]

    Ii = np.concatenate(rows_J + rows_S2 + rows_He)
    Jj = np.concatenate(cols_J + cols_S2 + cols_He)
    Vv = np.concatenate(vals_J + vals_S2 + vals_He)

    A = sp.csc_matrix((Vv, (Ii, Jj)), shape=(3 * N, 3 * N))

    b = np.empty(3 * N, dtype=float)
    b[:N] = T_J
    b[N : 2 * N] = T_S2 + gS2
    b[2 * N] = T_in
    b[2 * N + 1 :] = T_He[1:]

    return A, b


def sim_TH_2way(mdot, p, G, ACk, save_hist=False):
    """
    SIM_TH_2WAY  2-way coupled EM<->TH simulation.
     - Aggiorna Jc(B,T), R(T), q_AC(B,T) ad ogni dt
     - Risolve il circuito DP con directSolve a ogni dt
     - Risolve il sistema TH 3-eq (BDF1+upwind) a ogni dt
     - Termina early se T_S2 >> T_cs (quench)
    """
    N = 200
    dx = G.L_cable / (N - 1)
    xi = np.linspace(0.0, G.L_cable, N)

    # Interpolate per-turn quantities to fine grid (nearest neighbour by spira)
    f_nearest = interp1d(
        G.xi_centroid,
        np.arange(G.N_sp_dp, dtype=int),
        kind="nearest",
        fill_value="extrapolate",
    )
    spira_at_xi = f_nearest(xi).astype(int)
    spira_counts = np.bincount(spira_at_xi, minlength=G.N_sp_dp).astype(float)

    B_eff_x = G.B_eff_per_turn[spira_at_xi]
    dBperp_x = G.dBperp_dt[spira_at_xi]
    dBmod_x = G.dBmod_dt[spira_at_xi]
    T_cs_x = G.T_cs_per_turn[spira_at_xi]
    R_DP_x = G.R_DP[spira_at_xi]

    # CFL-based dt
    rho_He0, _, _, _ = props_He(G.T_in)
    v_He0 = mdot / (float(rho_He0) * G.A_He)
    dt = 0.5 * dx / v_He0
    n_steps = int(math.ceil(G.t_ramp / dt))
    dt = G.t_ramp / n_steps

    # IC
    T_J = np.full(N, G.T_in, dtype=float)
    T_S2 = np.full(N, G.T_in, dtype=float)
    T_He = np.full(N, G.T_in, dtype=float)

    # History
    quench_flag = False
    quench_t = np.nan
    quench_xi = np.nan
    history = None

    if save_hist:
        n_snap_t = 101
        n_save_int = max(1, n_steps // (n_snap_t - 1))
        n_snap = (n_steps // n_save_int) + 1
        history = ProjectStruct(
            t_arr=np.arange(n_steps + 1, dtype=float) * dt,
            t_hist=np.zeros(n_snap, dtype=float),
            T_J_hist=np.zeros((N, n_snap), dtype=float),
            T_S2_hist=np.zeros((N, n_snap), dtype=float),
            T_He_hist=np.zeros((N, n_snap), dtype=float),
            qAC_hist=np.zeros((N, n_snap), dtype=float),
            Tmax_hist=np.zeros(n_steps + 1, dtype=float),
            P_hyst_t=np.zeros(n_steps + 1, dtype=float),
            P_eddy_t=np.zeros(n_steps + 1, dtype=float),
            P_ohm_t=np.zeros(n_steps + 1, dtype=float),
            P_coup_t=np.zeros(n_steps + 1, dtype=float),
        )
        history.Tmax_hist[0] = float(np.max(T_S2))
        isnap = 0
        history.t_hist[isnap] = 0.0
        history.T_J_hist[:, isnap] = T_J
        history.T_S2_hist[:, isnap] = T_S2
        history.T_He_hist[:, isnap] = T_He

    min_margin_running = np.inf
    min_margin_profile = np.full(N, np.inf, dtype=float)

    # Accumulators di energia depositata sulla rampa (per bilancio)
    E_hyst_tot = 0.0
    E_eddy_tot = 0.0
    E_ohm_tot = 0.0
    E_coup_tot = 0.0
    dV_node = G.A_S2 * dx

    ACk_now = ProjectStruct(**ACk.to_dict())

    for n in range(1, n_steps + 1):
        t = n * dt

        # --- (a) Profilo di corrente e campo magnetico istantaneo ---
        if t <= G.t_ramp + 1e-12:
            I_t = ACk.I_op * (t / G.t_ramp)
            dIdt_now = ACk.dIdt
        else:
            I_t = ACk.I_op
            dIdt_now = 0.0

        scale_I = I_t / ACk.I_op
        B_eff_now = B_eff_x * scale_I
        dBperp_now = dBperp_x * (dIdt_now / ACk.dIdt)
        dBmod_now = dBmod_x * (dIdt_now / ACk.dIdt)

        ACk_now.I_op = I_t

        # --- (b) Dynamic AC losses (Hyst + Eddy + Ohmic + Coupling) at (B(t), T_S2^n) ---
        q_AC, q_hyst, q_eddy, q_ohm, q_coup, R_per_x = calc_losses_full(
            B_eff_now, dBperp_now, dBmod_now, T_S2, R_DP_x, ACk_now
        )

        # --- Bilancio energetico (accumulo integrale temporale) ---
        E_hyst_tot += float(np.sum(q_hyst)) * dV_node * dt
        E_eddy_tot += float(np.sum(q_eddy)) * dV_node * dt
        E_ohm_tot += float(np.sum(q_ohm)) * dV_node * dt
        E_coup_tot += float(np.sum(q_coup)) * dV_node * dt

        if save_hist:
            history.P_hyst_t[n] = float(np.sum(q_hyst)) * dV_node
            history.P_eddy_t[n] = float(np.sum(q_eddy)) * dV_node
            history.P_ohm_t[n] = float(np.sum(q_ohm)) * dV_node
            history.P_coup_t[n] = float(np.sum(q_coup)) * dV_node

        # --- (c) CIRCUIT SOLVER (directSolve) -----------------------
        R_per_turn_now = np.bincount(spira_at_xi, weights=R_per_x, minlength=G.N_sp_dp) / spira_counts
        solve_circuit_directSolve(
            R_per_turn_now, G.L_dp, G.M_ext_per_turn, I_t, dIdt_now
        )

        # --- (d) Lagged properties for TH solver (Eq. 2.1 - 2.3 + Gnielinski) ---
        rho_J, cp_J, k_J = props_SS(T_J)
        rho_S2, cp_S2, k_S2 = props_S2(T_S2, G.fracS2)
        rho_He, cp_He, k_He, mu_He = props_He(T_He)

        v_He = mdot / (rho_He * G.A_He)
        Re = rho_He * v_He * G.D_h / mu_He
        Pr = cp_He * mu_He / k_He

        f = (0.790 * np.log(Re) - 1.64) ** (-2.0)
        Nu = ((f / 8.0) * (Re - 1000.0) * Pr) / (
            1.0 + 12.7 * np.sqrt(f / 8.0) * ((Pr ** (2.0 / 3.0)) - 1.0)
        )
        h_conv = Nu * k_He / G.D_h

        G_S2_He = h_conv * G.P_S2_He

        rcp_J = rho_J * cp_J
        rcp_S2 = rho_S2 * cp_S2
        rcp_He = rho_He * cp_He

        aJ = k_J * dt / (rcp_J * (dx ** 2))
        aS2 = k_S2 * dt / (rcp_S2 * (dx ** 2))

        bJ = np.zeros(N, dtype=float)
        bS2J = np.zeros(N, dtype=float)
        bS2He = G_S2_He * dt / (rcp_S2 * G.A_S2)
        bHeS2 = G_S2_He * dt / (rcp_He * G.A_He)

        C = v_He * dt / dx
        gS2 = q_AC * dt / rcp_S2

        # Solve 3-equation system
        A_mat, b_vec = build_3eq_matrix(
            T_J, T_S2, T_He, aJ, aS2, bJ, bS2J, bS2He, bHeS2, C, gS2, G.T_in, N
        )
        x = spla.spsolve(A_mat, b_vec)
        T_J = x[:N]
        T_S2 = x[N : 2 * N]
        T_He = x[2 * N : 3 * N]

        # --- (e) Quench check ---------------------------------------
        Ic_at_B_now = ACk.Ic_0T_op / ((1.0 + B_eff_now / ACk.B0_op) ** ACk.alpha_op)
        ratio_t = (I_t / ACk.N_tapes) / Ic_at_B_now
        T_cs_now = ACk.Tc - (ACk.Tc - ACk.T_op) * (ratio_t ** (1.0 / ACk.s_scaling))
        T_cs_now[(T_cs_now < 0.0) | (T_cs_now > ACk.Tc)] = np.nan

        margin_now = T_cs_now - T_S2
        min_now = float(np.nanmin(margin_now))
        min_margin_running = min(min_margin_running, min_now)
        min_margin_profile = np.minimum(min_margin_profile, margin_now)

        if min_now < -5.0:
            qi = int(np.nanargmin(margin_now))
            quench_flag = True
            quench_t = t
            quench_xi = float(xi[qi])
            if not save_hist:
                return min_margin_running, quench_flag, None, None
            else:
                history.Tmax_hist[n] = float(np.max(T_S2))
                if isnap + 1 < n_snap:
                    isnap += 1
                    history.t_hist[isnap] = t
                    history.T_J_hist[:, isnap] = T_J
                    history.T_S2_hist[:, isnap] = T_S2
                    history.T_He_hist[:, isnap] = T_He
                    history.qAC_hist[:, isnap] = q_AC
                break

        # --- (f) History --------------------------------------------
        if save_hist:
            history.Tmax_hist[n] = float(np.max(T_S2))
            if (n % n_save_int == 0) and (isnap + 1 < n_snap):
                isnap += 1
                history.t_hist[isnap] = t
                history.T_J_hist[:, isnap] = T_J
                history.T_S2_hist[:, isnap] = T_S2
                history.T_He_hist[:, isnap] = T_He
                history.qAC_hist[:, isnap] = q_AC

    min_margin = min_margin_running
    if save_hist:
        result = ProjectStruct(
            T_J=T_J,
            T_S2=T_S2,
            T_He=T_He,
            T_cs=T_cs_x,
            xi=xi,
            quench_flag=quench_flag,
            quench_t=quench_t,
            quench_xi=quench_xi,
            min_margin_profile=min_margin_profile,
            E_hyst=E_hyst_tot,
            E_eddy=E_eddy_tot,
            E_ohm=E_ohm_tot,
            E_coup=E_coup_tot,
            E_tot=E_hyst_tot + E_eddy_tot + E_ohm_tot + E_coup_tot,
        )
        history.t_hist = history.t_hist[: isnap + 1]
        history.T_J_hist = history.T_J_hist[:, : isnap + 1]
        history.T_S2_hist = history.T_S2_hist[:, : isnap + 1]
        history.T_He_hist = history.T_He_hist[:, : isnap + 1]
        history.qAC_hist = history.qAC_hist[:, : isnap + 1]
        return min_margin, quench_flag, result, history

    return min_margin, quench_flag, None, None


# =========================================================================
#                         MAIN SCRIPT EXECUTION
# =========================================================================

def _load_magnetic_data(script_dir):
    mat_path = os.path.join(script_dir, "MagneticData_Final_nuovissima.mat")
    if not os.path.exists(mat_path):
        return run_step2_magnetic_design(show_plots=False)

    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    mag = ProjectStruct(
        N_sp_dp=int(data["N_sp_dp"]),
        num_turns=int(data["num_turns"]),
        N_pancakes=int(data["N_pancakes"]),
        H_active=float(data["H_active"]),
        R_DP=np.asarray(data["R_DP"], dtype=float).ravel(),
        Z_DP=np.asarray(data["Z_DP"], dtype=float).ravel(),
        L_per_turn=np.asarray(data["L_per_turn"], dtype=float).ravel(),
        xi_centroid=np.asarray(data["xi_centroid"], dtype=float).ravel(),
        L_cable_DP=float(data["L_cable_DP"]),
        B_perp_per_turn=np.asarray(data["B_perp_per_turn"], dtype=float).ravel(),
        B_par_per_turn=np.asarray(data["B_par_per_turn"], dtype=float).ravel(),
        B_mod_per_turn=np.asarray(data["B_mod_per_turn"], dtype=float).ravel(),
        B_eff_per_turn=np.asarray(data["B_eff_per_turn"], dtype=float).ravel(),
        L_dp=np.asarray(data["L_dp"], dtype=float),
        M_ext_per_turn=np.asarray(data["M_ext_per_turn"], dtype=float).ravel(),
    )
    return mag


def run_step6_optimus_prime(show_plots=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = setupProject_nuovissima()
    geom, tape = _load_cicc_geometry(script_dir)

    # Globals
    R_in = p.R_in
    H_sol_max = p.H_sol_max
    I_op = p.I_op
    B_target = p.B_target
    mu0 = p.mu0
    t_ramp = I_op / p.dIdt
    width_cicc = geom.width_cicc
    R_slot = geom.R_slot
    a_wire = geom.a_wire
    n_channels = geom.n_channels
    margin_req = p.dT_margin
    T_in = 20.0
    T_baseline = 20

    # %% =====================================================================
    #  SETUP GEOMETRIA E DATI MAGNETICI (Importati da Step 2)
    #  =====================================================================
    mag = _load_magnetic_data(script_dir)
    cprintf(
        True,
        "\nDati magnetici caricati: %d spire, H_active = %.3f m\n",
        mag.num_turns,
        mag.H_active,
    )

    # 1. Ricostruzione dell'asse spaziale del cavo (lunghezza srotolata)
    L_per_turn = 2.0 * np.pi * mag.R_DP
    xi_centroid = np.cumsum(L_per_turn) - L_per_turn / 2.0
    L_cable_DP = float(np.sum(L_per_turn))

    # =========================================================================
    # 2. Calcolo Aree (MODELLO 2 COMPONENTI: Cu + SC)
    # =========================================================================
    A_SC_total = geom.N_tapes * p.w_tape * p.thick_sc_tape
    A_Cu_tape = geom.N_tapes * p.w_tape * p.thick_cu_tape
    A_Cu_net = (
        np.pi * (geom.R_former ** 2)
        - np.pi * (geom.R_central ** 2)
        - n_channels * geom.w_slot * geom.h_slot
    )
    A_Cu_total = A_Cu_net + A_Cu_tape

    # Conduttore S2 = Cu + SC
    A_S2_total = A_Cu_total + A_SC_total

    A_J_total = (width_cicc ** 2) - ((2.0 * geom.R_former) ** 2)
    A_slot_he = geom.w_slot * geom.h_slot - geom.tapes_per_slot * p.w_tape * p.thick_tape
    A_He_total = np.pi * (geom.R_central ** 2) + n_channels * A_slot_he

    # Frazioni d'area per le proprietà equivalenti (Eq. 2.26 - 2.27)
    fracS2 = ProjectStruct(
        Cu=A_Cu_total / A_S2_total,
        SC=A_SC_total / A_S2_total,
    )

    # 3. Struttura delle costanti fisiche (ACk)
    ACk = ProjectStruct(
        E_c=1e-4,
        n_pl=25,
        w_tape=p.w_tape,
        thick_tape=p.thick_tape,
        Tc=tape.Tc,
        s_scaling=tape.s_T,
        B0_op=tape.B0_op,
        alpha_op=tape.alpha_op,
        J_c0_base=tape.J_c0_base,
        R_former=geom.R_former,
        RRR_cu=p.RRR_cu,
        A_REBCO_geom=A_SC_total,
        A_Cu_net=A_Cu_net,
        A_S2=A_S2_total,
        A_REBCO_film=A_SC_total,
        A_Cu_total=A_Cu_total,
        w_hyst_eff=p.w_tape / p.n_filaments,
        I_op=I_op,
        N_tapes=geom.N_tapes,
        t_ramp=t_ramp,
        dIdt=p.dIdt,
        tau_coupling=tape.tau_coupling,
        Ic_0T_op=tape.Ic_0T_op,
        T_op=p.T_op,
        r_contact_JS2=p.r_contact_JS2,
        thickness_ss=geom.thickness_ss,
        fracS2=fracS2,
    )

    cprintf(True, "  L_cable = %.2f m pronta per simulazione TH.\n", L_cable_DP)

    # %% =====================================================================
    #  PHASE 0 : BASELINE DIAGNOSTICS @ T = 20 K
    #  =====================================================================
    cprintf(
        True,
        "\n============== PHASE 0 : Baseline diagnostics (T=20K) ==============\n",
    )

    T_S2_base = T_baseline * np.ones(mag.N_sp_dp, dtype=float)
    dBperp_base = mag.B_perp_per_turn / t_ramp
    dBmod_base = mag.B_mod_per_turn / t_ramp
    dBpar_base = mag.B_par_per_turn / t_ramp

    q_AC_b, q_hyst_b, q_eddy_b, q_ohm_b, q_coup_b, R_per_turn_b = calc_losses_full(
        mag.B_eff_per_turn, dBperp_base, dBmod_base, T_S2_base, mag.R_DP, ACk
    )

    P_hyst_lin = q_hyst_b * A_S2_total
    P_eddy_lin = q_eddy_b * A_S2_total
    P_ohm_lin = q_ohm_b * A_S2_total
    P_coup_lin = q_coup_b * A_S2_total
    P_tot_lin = P_hyst_lin + P_eddy_lin + P_ohm_lin + P_coup_lin

    E_hyst_static = float(np.sum(P_hyst_lin * L_per_turn) * t_ramp)
    E_eddy_static = float(np.sum(P_eddy_lin * L_per_turn) * t_ramp)
    E_ohm_static = float(np.sum(P_ohm_lin * L_per_turn) * t_ramp)
    E_coup_static = float(np.sum(P_coup_lin * L_per_turn) * t_ramp)
    E_tot_static = E_hyst_static + E_eddy_static + E_ohm_static + E_coup_static

    cprintf(True, "  Stima statica energia per DP @ T=20K (perdite a B_peak):\n")
    cprintf(True, "    Hyst       : %.2f kJ\n", E_hyst_static / 1e3)
    cprintf(True, "    Eddy Cu    : %.2f kJ\n", E_eddy_static / 1e3)
    cprintf(True, "    Ohmic SC   : %.2f kJ\n", E_ohm_static / 1e3)
    cprintf(True, "    Coupling   : %.2f kJ\n", E_coup_static / 1e3)
    cprintf(True, "    TOTALE     : %.2f kJ\n", E_tot_static / 1e3)

    turn_idx = np.arange(1, mag.N_sp_dp + 1)

    # --- Fig P01 - Stacked area: loss breakdown per spira -----------------
    plt.figure(num="P01 - Baseline Thermal: loss breakdown", figsize=(9.5, 4.6))
    plt.stackplot(
        turn_idx,
        P_hyst_lin,
        P_eddy_lin,
        P_coup_lin,
        P_ohm_lin,
        labels=["Hysteresis REBCO", "Eddy Cu former", "Coupling inter-tape", "Ohmic SC"],
    )
    plt.axvline(mag.num_turns + 0.5, color="g", linestyle="--", linewidth=1.5, label="P1|P2")
    plt.grid(True)
    plt.xlabel("Indice spira")
    plt.ylabel("Potenza lineare [W/m]")
    plt.legend(loc="best")
    plt.title(
        f"Loss breakdown per spira @ T={T_baseline}K, $B_{{peak}}$ ($P_{{tot}}$ = {np.sum(P_tot_lin * L_per_turn):.1f} W)"
    )
    plt.tight_layout()

    # --- Fig P02 - dB/dt per spira ---------------------------------------
    plt.figure(num="P02 - Field rate per spira", figsize=(9.0, 4.2))
    plt.plot(turn_idx, dBperp_base, "r-s", linewidth=2, label=r"$dB_{perp}/dt$")
    plt.plot(turn_idx, dBpar_base, "b-o", linewidth=2, label=r"$dB_{par}/dt$")
    plt.plot(turn_idx, dBmod_base, "k-^", linewidth=1.5, label=r"$d|B|/dt$")
    plt.axvline(mag.num_turns + 0.5, color="g", linestyle="--", label="P1|P2")
    plt.grid(True)
    plt.xlabel("Indice spira")
    plt.ylabel("dB/dt [T/s]")
    plt.legend(loc="best")
    plt.title("Field rate per spira (costante in t, scala con dI/dt)")
    plt.tight_layout()

    # --- Fig P05 - Confronto perdite per spira (NON sommate, scala log) ----
    plt.figure(num="P05 - Loss comparison per turn (log)", figsize=(9.5, 4.6))
    P_floor = 1e-6
    plt.semilogy(
        turn_idx, P_hyst_lin + P_floor, "r-s", linewidth=2, markersize=5, label="Hysteresis REBCO"
    )
    plt.semilogy(
        turn_idx, P_eddy_lin + P_floor, "b-o", linewidth=2, markersize=5, label="Eddy Cu former"
    )
    plt.semilogy(
        turn_idx, P_ohm_lin + P_floor, "m-d", linewidth=2, markersize=5, label="Ohmic SC"
    )
    plt.semilogy(
        turn_idx, P_coup_lin + P_floor, "g-^", linewidth=2, markersize=5, label="Coupling inter-tape"
    )
    plt.axvline(mag.num_turns + 0.5, color="k", linestyle="--", linewidth=1.2)
    plt.grid(True)
    plt.xlabel("Indice spira")
    plt.ylabel("Potenza lineare [W/m]")
    plt.legend(loc="best")
    plt.title(f"Confronto perdite AC per spira @ T={T_baseline}K, $B_{{peak}}$ (scala log)")
    plt.tight_layout()

    # --- Fig P03 - Heatmap della matrice induttanze L_dp ------------------
    plt.figure(num="P03 - L_dp heatmap", figsize=(5.8, 5.2))
    im = plt.imshow(
        mag.L_dp * 1e6,
        extent=[0.5, mag.N_sp_dp + 0.5, mag.N_sp_dp + 0.5, 0.5],
        aspect="equal",
    )
    cb = plt.colorbar(im)
    cb.set_label(r"$M_{ij}$ [$\mu$H]")
    plt.xlabel("j (spira)")
    plt.ylabel("i (spira)")
    plt.title(f"Matrice mutue induttanze $L_{{dp}}$ ({mag.N_sp_dp}x{mag.N_sp_dp})")
    plt.axvline(mag.num_turns + 0.5, color="w", linestyle="--", linewidth=1.5)
    plt.axhline(mag.num_turns + 0.5, color="w", linestyle="--", linewidth=1.5)
    plt.tight_layout()

    # --- Fig P04 - Mappa |B|(r,z) sul piano del solenoide ----------------
    cprintf(True, "  Calcolo mappa |B|(r,z) per visualizzazione ...\n")
    r_grid = np.linspace(0.001, p.R_in + mag.num_turns * width_cicc + 0.05, 60)
    z_grid = np.linspace(-mag.H_active / 2.0 * 1.05, mag.H_active / 2.0 * 1.05, 80)
    RRg, ZZg = np.meshgrid(r_grid, z_grid)

    pitch_R = width_cicc + 1e-3
    r_layers0 = p.R_in + np.arange(mag.num_turns, dtype=float) * pitch_R + width_cicc / 2.0
    pitch_DP = 2.0 * width_cicc + 1e-3 + 2e-3
    N_DP = mag.N_pancakes // 2
    Z_DP_c = (np.arange(1, N_DP + 1) - (N_DP + 1) / 2.0) * pitch_DP
    z_rel = np.array([+(width_cicc + 1e-3) / 2.0, -(width_cicc + 1e-3) / 2.0])
    z_pks0 = (Z_DP_c[:, np.newaxis] + z_rel[np.newaxis, :]).ravel(order="C")

    Ki_g, Pi_g = np.meshgrid(
        np.arange(mag.num_turns), np.arange(mag.N_pancakes), indexing="ij"
    )
    R_src_v = r_layers0[Ki_g.ravel(order="F")]
    Z_src_v = z_pks0[Pi_g.ravel(order="F")]
    I_src_v = np.full(R_src_v.size, I_op, dtype=float)
    Brg, Bzg = biot_savart_loops_vec(
        R_src_v, Z_src_v, I_src_v, RRg.ravel(), ZZg.ravel(), mu0
    )
    Bmod_grid = np.reshape(np.sqrt(Brg ** 2 + Bzg ** 2), RRg.shape)

    plt.figure(num="P04 - Mappa |B|(r,z)", figsize=(7.6, 5.2))
    cf = plt.contourf(RRg, ZZg, Bmod_grid, 25, cmap="viridis")
    cb = plt.colorbar(cf)
    cb.set_label("|B| [T]")
    plt.plot(
        mag.R_DP,
        mag.Z_DP,
        "wo",
        markerfacecolor="w",
        markersize=4,
        label="DP equatoriale",
    )
    plt.axvline(p.R_in, color="w", linestyle="-.", linewidth=1)
    plt.axvline(p.R_in + mag.num_turns * width_cicc, color="w", linestyle="-.", linewidth=1)
    plt.axhline(+mag.H_active / 2.0, color="w", linestyle="-.", linewidth=1)
    plt.axhline(-mag.H_active / 2.0, color="w", linestyle="-.", linewidth=1)
    plt.xlabel("r [m]")
    plt.ylabel("z [m]")
    plt.title(
        f"|B| nel piano (r,z), max = {np.max(Bmod_grid):.2f} T  ($B_{{target}}$ = {p.B_target:.0f} T)"
    )
    plt.tight_layout()

    # %% =====================================================================
    #  PHASE 1 : BISEZIONE FINE PER MDOT MINIMO
    #  =====================================================================
    cprintf(True, "\n============== PHASE 1 : BISEZIONE MDOT ==============\n")

    P_J_S2 = 2.0 * np.pi * geom.R_former
    P_S2_He = 2.0 * np.pi * geom.R_central + n_channels * 2.0 * (geom.w_slot + geom.h_slot)
    D_h = 4.0 * A_He_total / P_S2_He
    I_per_tape = I_op / geom.N_tapes

    Ic_at_B_per_turn = tape.Ic_0T_op / (
        (1.0 + mag.B_eff_per_turn / tape.B0_op) ** tape.alpha_op
    )
    ratio = I_per_tape / Ic_at_B_per_turn
    T_cs_per_turn = tape.Tc - (tape.Tc - p.T_op) * (ratio ** (1.0 / tape.s_T))
    T_cs_per_turn[(T_cs_per_turn < 0.0) | (T_cs_per_turn > tape.Tc)] = np.nan

    G = ProjectStruct(
        N_sp_dp=mag.N_sp_dp,
        num_turns=mag.num_turns,
        N_pancakes=mag.N_pancakes,
        R_DP=mag.R_DP,
        Z_DP=mag.Z_DP,
        B_perp_per_turn=mag.B_perp_per_turn,
        B_par_per_turn=mag.B_par_per_turn,
        B_mod_per_turn=mag.B_mod_per_turn,
        B_eff_per_turn=mag.B_eff_per_turn,
        dBperp_dt=mag.B_perp_per_turn / t_ramp,
        dBmod_dt=mag.B_mod_per_turn / t_ramp,
        L_dp=mag.L_dp,
        M_ext_per_turn=mag.M_ext_per_turn,
        L_per_turn=L_per_turn,
        xi_centroid=xi_centroid,
        L_cable=L_cable_DP,
        A_REBCO=A_SC_total,
        A_Cu_net=A_Cu_net,
        A_S2=A_S2_total,
        A_J=A_J_total,
        A_He=A_He_total,
        R_former=geom.R_former,
        thickness_ss=geom.thickness_ss,
        r_contact_JS2=p.r_contact_JS2,
        P_J_S2=P_J_S2,
        P_S2_He=P_S2_He,
        D_h=D_h,
        T_in=T_in,
        margin_req=margin_req,
        t_ramp=t_ramp,
        I_per_tape=I_per_tape,
        fracS2=fracS2,
        T_cs_per_turn=T_cs_per_turn,
    )

    cprintf(
        True,
        "  T_cs per turn: [%.1f, %.1f] K\n",
        np.nanmin(T_cs_per_turn),
        np.nanmax(T_cs_per_turn),
    )

    headroom_cold = float(np.nanmin(T_cs_per_turn) - T_in - margin_req)
    lf_max = float(np.max(G.I_per_tape / Ic_at_B_per_turn))
    if headroom_cold < 2.0:
        verdict = "(STRETTO: mdot_min alta; abbassa f_load_design per slack)"
    elif headroom_cold > 12.0:
        verdict = "(LASCO: mdot_min ~floor; alza f_load_design per un punto utile)"
    else:
        verdict = "(ok)"
    cprintf(
        True,
        "  [Diagnostica] load_frac worst = %.2f, headroom = %.1f K %s\n",
        lf_max,
        headroom_cold,
        verdict,
    )

    # Bracket iniziale
    mdot_low = 1e-3
    mdot_high = 15e-3
    mdot_tol = 0.1e-3
    cprintf(
        True,
        "\n  Test bracket [%.1f, %.1f] g/s ...\n",
        mdot_low * 1e3,
        mdot_high * 1e3,
    )
    mar_low, qf_low, _, _ = sim_TH_2way(mdot_low, p, G, ACk, False)
    mar_high, qf_high, _, _ = sim_TH_2way(mdot_high, p, G, ACk, False)
    cprintf(
        True,
        "    mdot = %5.2f g/s -> margine = %6.2f K %s\n",
        mdot_low * 1e3,
        mar_low,
        q2str(qf_low),
    )
    cprintf(
        True,
        "    mdot = %5.2f g/s -> margine = %6.2f K %s\n",
        mdot_high * 1e3,
        mar_high,
        q2str(qf_high),
    )

    mdot_plot_hist = [mdot_low, mdot_high]
    mar_plot_hist = [mar_low, mar_high]

    if mar_high < margin_req:
        cprintf(
            True,
            "\n  *** DESIGN INFEASIBLE: anche mdot_max non passa margine req. ***\n",
        )
        mdot_min = mdot_high
    elif mar_low >= margin_req:
        cprintf(
            True,
            "\n  Anche mdot_low passa: mdot_min < %.1f g/s\n",
            mdot_low * 1e3,
        )
        mdot_min = mdot_low
    else:
        cprintf(True, "\n  Bisezione (tol = %.2f g/s):\n", mdot_tol * 1e3)
        cprintf(True, "  %6s  %9s  %8s  %s\n", "iter", "mdot[g/s]", "mar[K]", "verd")
        it = 0
        while (mdot_high - mdot_low) > mdot_tol and it < 20:
            it += 1
            mdot_mid = 0.5 * (mdot_low + mdot_high)
            mar_mid, qf_mid, _, _ = sim_TH_2way(mdot_mid, p, G, ACk, False)

            mdot_plot_hist.append(mdot_mid)
            mar_plot_hist.append(mar_mid)

            verd = "PASS" if mar_mid >= margin_req else "FAIL"
            cprintf(
                True,
                "  %6d  %9.3f  %8.2f  %s%s\n",
                it,
                mdot_mid * 1e3,
                mar_mid,
                verd,
                q2str(qf_mid),
            )
            if mar_mid >= margin_req:
                mdot_high = mdot_mid
            else:
                mdot_low = mdot_mid
        mdot_min = mdot_high

    cprintf(
        True,
        "\n  >>>> mdot_min = %.3f g/s (%.1f%% capacita) <<<<\n",
        mdot_min * 1e3,
        100.0 * mdot_min / p.mdot_max,
    )

    # %% =====================================================================
    #  PHASE 2 : FULL 2-WAY COUPLED TRANSIENT @ mdot_min
    #  =====================================================================
    cprintf(
        True,
        "\n============== PHASE 2 : Final transient @ mdot_min ==============\n",
    )

    margin_final, quench_flag_final, result, history = sim_TH_2way(
        mdot_min, p, G, ACk, True
    )

    xi = result.xi

    # Fig 6 - T(xi) at end of ramp
    idx_ramp = int(np.argmin(np.abs(history.t_hist - G.t_ramp)))
    T_J_ramp = history.T_J_hist[:, idx_ramp]
    T_S2_ramp = history.T_S2_hist[:, idx_ramp]
    T_He_ramp = history.T_He_hist[:, idx_ramp]

    plt.figure(num="06 - Final T(xi)", figsize=(10.0, 4.6))
    plt.plot(xi, T_J_ramp, "b-", linewidth=2, label=r"$T_J$")
    plt.plot(xi, T_S2_ramp, "r-", linewidth=2, label=r"$T_{S2}$")
    plt.plot(xi, T_He_ramp, "k--", linewidth=2, label=r"$T_{He}$")
    plt.plot(xi, result.T_cs, "g:", linewidth=1.8, label=r"$T_{cs}$")
    plt.axhline(T_in, color="m", linestyle="--", label=r"$T_{in}$")
    plt.axvline(L_cable_DP / 2.0, color=(0.5, 0.5, 0.5), linestyle="--")
    plt.grid(True)
    plt.xlabel(r"$\xi$ [m]")
    plt.ylabel("T [K]")
    plt.legend(loc="upper left")
    plt.title(
        f"Profili a fine rampa (t = {G.t_ramp:.0f} s) @ $mdot_{{min}}$ = {mdot_min*1e3:.2f} g/s"
    )
    plt.tight_layout()

    # Fig 7 - Tmax(t)
    plt.figure(num="07 - Tmax history", figsize=(8.0, 4.2))
    plt.plot(history.t_arr, history.Tmax_hist, "r-", linewidth=2)
    plt.axhline(T_in, color="m", linestyle="--", label=r"$T_{in}$")
    plt.axhline(
        T_in + margin_req,
        color="g",
        linestyle="--",
        label=f"$T_{{in}}$+{margin_req:.0f}K",
    )
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel(r"max $T_{S2}$ [K]")
    plt.title(r"Evoluzione di $T_{S2,max}$ durante la rampa")
    plt.legend(loc="best")
    plt.tight_layout()

    # Fig 8 - T_S2 waterfall
    plt.figure(num="08 - T_S2(xi,t)", figsize=(9.0, 4.2))
    im8 = plt.imshow(
        history.T_S2_hist,
        extent=[history.t_hist[0], history.t_hist[-1], xi[0], xi[-1]],
        origin="lower",
        aspect="auto",
        cmap="hot",
    )
    cb8 = plt.colorbar(im8)
    cb8.set_label(r"$T_{S2}$ [K]")
    plt.xlabel("t [s]")
    plt.ylabel(r"$\xi$ [m]")
    plt.title(r"$T_{S2}(\xi,t)$")
    plt.tight_layout()

    # Fig 9 - q_AC dynamic waterfall
    plt.figure(num="09 - q_AC(xi,t)", figsize=(9.0, 4.2))
    im9 = plt.imshow(
        history.qAC_hist / 1e3,
        extent=[history.t_hist[0], history.t_hist[-1], xi[0], xi[-1]],
        origin="lower",
        aspect="auto",
        cmap="hot",
    )
    cb9 = plt.colorbar(im9)
    cb9.set_label(r"$q_{AC}$ [kW/$m^3$]")
    plt.xlabel("t [s]")
    plt.ylabel(r"$\xi$ [m]")
    plt.title("Sorgente AC dinamica")
    plt.tight_layout()

    # Fig 10 - margin profile
    plt.figure(num="10 - Margin profile", figsize=(10.0, 3.8))
    plt.plot(xi, result.min_margin_profile, "b-", linewidth=2)
    plt.axhline(
        margin_req,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"req. = {margin_req:.0f} K",
    )
    plt.axvline(L_cable_DP / 2.0, color=(0.5, 0.5, 0.5), linestyle="--")
    plt.grid(True)
    plt.xlabel(r"$\xi$ [m]")
    plt.ylabel(r"$\Delta T_{mar}$ [K]")
    plt.title("Profilo del Margine Minimo Assoluto (su tutto il transitorio)")
    plt.legend(loc="best")
    plt.tight_layout()

    # Final report
    cprintf(True, "\n=============== STEP 6 FINAL RESULTS ===============\n")
    cprintf(True, "mdot_min       = %.3f g/s\n", mdot_min * 1e3)
    cprintf(True, "T_in           = %.1f K\n", T_in)
    cprintf(True, "max T_J        = %.2f K\n", float(np.max(result.T_J)))
    cprintf(True, "max T_S2       = %.2f K\n", float(np.max(result.T_S2)))
    cprintf(True, "T_He outlet    = %.2f K\n", float(result.T_He[-1]))
    cprintf(
        True,
        "min margin     = %.2f K (richiesto %.0f K)\n",
        margin_final,
        margin_req,
    )
    if quench_flag_final:
        cprintf(
            True,
            "*** QUENCH at t=%.1f s, xi=%.1f m ***\n",
            result.quench_t,
            result.quench_xi,
        )
    elif margin_final >= margin_req:
        cprintf(True, "Design OK. Margine termico rispettato.\n")
    else:
        cprintf(
            True,
            "*** WARNING: No Quench, ma Margine < 5 K. Design INADEGUATO ***\n",
        )
    cprintf(True, "====================================================\n")

    # --- Bilancio energetico integrato sulla rampa (per DP equatoriale) ---
    cprintf(True, "\n--- BILANCIO ENERGETICO sulla rampa (per DP) ---\n")
    cprintf(
        True,
        "  E_hyst (Brandt thin strip) : %8.2f kJ  (%5.1f %%)\n",
        result.E_hyst / 1e3,
        100.0 * result.E_hyst / result.E_tot,
    )
    cprintf(
        True,
        "  E_eddy (Cu former)         : %8.2f kJ  (%5.1f %%)\n",
        result.E_eddy / 1e3,
        100.0 * result.E_eddy / result.E_tot,
    )
    cprintf(
        True,
        "  E_ohm  (SC power-law)      : %8.2f kJ  (%5.1f %%)\n",
        result.E_ohm / 1e3,
        100.0 * result.E_ohm / result.E_tot,
    )
    cprintf(
        True,
        "  E_coup (inter-tape)        : %8.2f kJ  (%5.1f %%)\n",
        result.E_coup / 1e3,
        100.0 * result.E_coup / result.E_tot,
    )
    cprintf(True, "  -------------------------------------------------\n")
    cprintf(True, "  E_TOTAL (per DP equatoriale): %8.2f kJ\n", result.E_tot / 1e3)
    cprintf(
        True,
        "  Stima statica Phase 0      : %8.2f kJ  (delta dinamico %+.1f %%)\n",
        E_tot_static / 1e3,
        100.0 * (result.E_tot - E_tot_static) / E_tot_static,
    )

    dT_He_outlet = float(result.T_He[-1] - T_in)
    _, cp_He_avg, _, _ = props_He(0.5 * (T_in + float(result.T_He[-1])))
    E_He_carried = mdot_min * float(cp_He_avg) * dT_He_outlet * G.t_ramp
    cprintf(
        True,
        "  Energia portata via elio   : %8.2f kJ  (%.1f %% del totale)\n",
        E_He_carried / 1e3,
        100.0 * E_He_carried / result.E_tot,
    )
    cprintf(
        True,
        "  Energia nei solidi (S2+J)  : %8.2f kJ  (%.1f %% del totale)\n",
        (result.E_tot - E_He_carried) / 1e3,
        100.0 * (result.E_tot - E_He_carried) / result.E_tot,
    )
    cprintf(True, "====================================================\n")

    # Fig 11 - Margine di Temperatura vs Portata di Elio
    mdot_arr_gs = np.asarray(mdot_plot_hist, dtype=float) * 1e3
    mar_arr = np.asarray(mar_plot_hist, dtype=float)
    sort_idx = np.argsort(mdot_arr_gs)
    mdot_sorted = mdot_arr_gs[sort_idx]
    mar_sorted = mar_arr[sort_idx]

    plt.figure(num="11 - Margine vs Portata Elio", figsize=(8.5, 4.5))
    plt.plot(
        mdot_sorted,
        mar_sorted,
        "b-o",
        linewidth=2,
        markerfacecolor="b",
        markersize=5,
        label="Punti di Bisezione",
    )
    plt.axhline(
        margin_req,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Margine Richiesto = {margin_req:.0f} K",
    )
    plt.plot(
        mdot_min * 1e3,
        margin_final,
        "go",
        markersize=12,
        linewidth=3,
        markerfacecolor="g",
        label=r"Portata Minima Ottima ($\dot{m}_{min}$)",
    )
    plt.axvline(mdot_min * 1e3, color="g", linestyle=":", linewidth=1.8)
    plt.grid(True)
    plt.xlabel(r"Portata di Elio $\dot{m}$ [g/s]", fontsize=11)
    plt.ylabel(r"Margine di Temperatura Minimo $\Delta T_{mar}$ [K]", fontsize=11)
    plt.title(
        "Andamento del Margine Termico Minimo in funzione della Portata",
        fontsize=12,
    )
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    # Fig 12 - Evoluzione temporale delle perdite AC durante la rampa
    plt.figure(num="12 - Loss time evolution", figsize=(9.5, 4.6))
    P_floor = 1e-6
    plt.semilogy(
        history.t_arr,
        history.P_hyst_t + P_floor,
        "r-",
        linewidth=2.5,
        label="Hysteresis REBCO",
    )
    plt.semilogy(
        history.t_arr,
        history.P_eddy_t + P_floor,
        "b-",
        linewidth=2.5,
        label="Eddy Cu former",
    )
    plt.semilogy(
        history.t_arr,
        history.P_ohm_t + P_floor,
        "m-",
        linewidth=2.5,
        label="Ohmic SC",
    )
    plt.semilogy(
        history.t_arr,
        history.P_coup_t + P_floor,
        "g-",
        linewidth=2.5,
        label="Coupling inter-tape",
    )
    plt.grid(True)
    plt.xlabel("t [s]", fontsize=11)
    plt.ylabel("Potenza totale per DP [W]", fontsize=11)
    plt.legend(loc="best", fontsize=10)
    plt.title("Evoluzione temporale delle perdite AC durante la rampa", fontsize=12)
    plt.tight_layout()

    if show_plots:
        plt.show()

    return mdot_min, margin_final, result, history


if __name__ == "__main__":
    run_step6_optimus_prime(show_plots=True)
