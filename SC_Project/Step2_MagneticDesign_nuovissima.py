# =========================================================================
# Step2_MagneticDesign_nuovissima.py
# =========================================================================
# Master script integrato:
#   PHASE 0a : MAGNETIC MAPPING (Step 2 logic, Biot-Savart + L_dp via inductanceMatrix1d)
#   Salvataggio dati in MagneticData_Final_nuovissima.mat per lo Step 6
# =========================================================================
import os
import time
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.special import ellipk, ellipe
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from setupProject_nuovissima import setupProject_nuovissima, ProjectStruct
from cprintf import cprintf
from biot_savart_loops_vec import biot_savart_loops_vec
from segmentMagneticField1d import segmentMagneticField1d
from inductanceMatrix1d import inductanceMatrix1d
from dimensionamento_CICC_nuovissima import run_dimensionamento_CICC


def _load_cicc_geometry(script_dir):
    mat_path = os.path.join(script_dir, "CICC_Geometry_nuovissima.mat")
    if not os.path.exists(mat_path):
        return run_dimensionamento_CICC(show_plots=False)

    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    geom_raw = data["geom"]
    tape_raw = data["tape"]

    geom = ProjectStruct()
    for field in geom_raw._fieldnames:
        val = getattr(geom_raw, field)
        geom[field] = int(val) if field in ("N_tapes", "tapes_per_slot", "n_channels") else float(val)

    tape = ProjectStruct()
    for field in tape_raw._fieldnames:
        tape[field] = float(getattr(tape_raw, field))

    return geom, tape


def run_step2_magnetic_design(show_plots=True):
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
    #  PHASE 0a : MAGNETIC MAPPING (Step 2 logic, interno)
    #  =====================================================================
    cprintf(True, "\n============== PHASE 0a : MAGNETIC MAPPING ==============\n")

    # --- 1. Definizione dello Spazio dei Parametri ---
    gap_R = 0.001       # 1 mm distanza radiale TRA LE SPIRE
    gap_DP_int = 0.001  # 1 mm distanza assiale TRA I DUE PANCAKE di uno stesso DP
    gap_DP_ext = 0.002  # 2 mm distanza assiale TRA I DOUBLE PANCAKE

    # Vogliamo un numero DISPARI di Double Pancake (DP centrale a Z=0).
    N_DP_vec = np.arange(11, 32, 2)
    N_pancakes_vec = N_DP_vec * 2  # [22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62]
    num_configs = len(N_pancakes_vec)

    theta_slots = np.arange(n_channels, dtype=float) * (2.0 * np.pi / n_channels)

    # Vettori per salvare i risultati di ogni configurazione
    L_rebco_totale = np.zeros(num_configs, dtype=float)
    N_turns_req = np.zeros(num_configs, dtype=int)
    B_max_achieved = np.zeros(num_configs, dtype=float)

    cprintf(True, "\nAvvio Ottimizzazione Geometria (gap inclusi, N_DP dispari)...\n")

    # --- 2. CICLO ESTERNO (Scansione Assiale) ---
    for i in range(num_configs):
        N_pk_test = int(N_pancakes_vec[i])
        N_DP_test = N_pk_test // 2

        # Vettore z_pks (coordinate Z di tutti i pancake del solenoide)
        pitch_DP = 2.0 * width_cicc + gap_DP_int + gap_DP_ext
        Z_DP_centers = (np.arange(1, N_DP_test + 1) - (N_DP_test + 1) / 2.0) * pitch_DP

        # Quote relative ai centri per il Pancake 1 (sopra) e Pancake 2 (sotto)
        z_rel = np.array([+(width_cicc + gap_DP_int) / 2.0, -(width_cicc + gap_DP_int) / 2.0])
        z_pks_mat = Z_DP_centers[:, np.newaxis] + z_rel[np.newaxis, :]  # [N_DP_test x 2]
        z_pks = z_pks_mat.ravel(order="C")  # [N_pk_test]

        # H_active reale calcolata dai centri estremi
        H_active_test = np.max(z_pks) - np.min(z_pks) + width_cicc

        # --- 3. CICLO INTERNO (Scansione Radiale / Auto-tuning) ---
        num_turns_test = 1
        B_max_val = 0.0

        while B_max_val < B_target and num_turns_test < 200:
            pitch_R = width_cicc + gap_R
            r_layers = R_in + np.arange(num_turns_test, dtype=float) * pitch_R + width_cicc / 2.0

            # MATLAB: [Ki, Pi] = ndgrid(1:num_turns_test, 1:N_pk_test); Ki(:), Pi(:) -> Fortran order
            Ki, Pi = np.meshgrid(
                np.arange(num_turns_test), np.arange(N_pk_test), indexing="ij"
            )
            R_src = r_layers[Ki.ravel(order="F")]
            Z_src = z_pks[Pi.ravel(order="F")]
            I_src = np.full(R_src.size, I_op, dtype=float)

            # DP equatoriale centrale (essendo dispari, il centro è sempre a 0)
            z_P1 = +(width_cicc + gap_DP_int) / 2.0
            z_P2 = -(width_cicc + gap_DP_int) / 2.0
            R_DP = np.concatenate([r_layers[::-1], r_layers])
            Z_DP = np.concatenate(
                [np.full(num_turns_test, z_P1), np.full(num_turns_test, z_P2)]
            )

            SpG, SlG = np.meshgrid(
                np.arange(R_DP.size), np.arange(n_channels), indexing="ij"
            )
            th_eval = theta_slots[SlG.ravel(order="F")]
            R_obs = R_DP[SpG.ravel(order="F")] + R_slot * np.cos(th_eval)
            Z_obs = Z_DP[SpG.ravel(order="F")] + R_slot * np.sin(th_eval)

            Br, Bz = biot_savart_loops_vec(R_src, Z_src, I_src, R_obs, Z_obs, mu0)
            B_max_val = float(np.max(np.sqrt(Br ** 2 + Bz ** 2)))

            if B_max_val < B_target:
                num_turns_test += 1

        # --- 4. Calcolo della Funzione Obiettivo (Lunghezza Nastro) ---
        L_cavo_DP = np.sum(2.0 * np.pi * R_DP)
        L_cavo_tot = L_cavo_DP * (N_pk_test / 2.0)
        L_rebco_totale[i] = L_cavo_tot * geom.N_tapes

        N_turns_req[i] = num_turns_test
        B_max_achieved[i] = B_max_val

    # --- 5. Selezione dell'Ottimo ---
    idx_opt = int(np.argmin(L_rebco_totale))
    min_rebco = float(L_rebco_totale[idx_opt])

    N_pancakes_opt = int(N_pancakes_vec[idx_opt])
    num_turns_opt = int(N_turns_req[idx_opt])
    H_active_opt = N_pancakes_opt * width_cicc
    B_max_val = float(B_max_achieved[idx_opt])

    cprintf(True, "  Ottimizzazione conclusa. Design Scelto:\n")
    cprintf(
        True,
        "  Pancake: %d, Spire: %d, REBCO Tot: %.1f km\n\n",
        N_pancakes_opt,
        num_turns_opt,
        min_rebco / 1000.0,
    )

    # >>> DA QUI IN POI, ASSEGNI LE VARIABILI OTTIME <<<
    N_pancakes = N_pancakes_opt
    N_DP_opt = N_pancakes // 2
    num_turns = num_turns_opt
    H_active = H_active_opt
    N_sp_dp = 2 * num_turns

    # Ricalcolo rigoroso dei vettori geometrici per la configurazione OTTIMA
    pitch_R = width_cicc + gap_R
    r_layers = R_in + np.arange(num_turns, dtype=float) * pitch_R + width_cicc / 2.0

    pitch_DP = 2.0 * width_cicc + gap_DP_int + gap_DP_ext
    Z_DP_centers = (np.arange(1, N_DP_opt + 1) - (N_DP_opt + 1) / 2.0) * pitch_DP
    z_rel = np.array([+(width_cicc + gap_DP_int) / 2.0, -(width_cicc + gap_DP_int) / 2.0])
    z_pks_mat = Z_DP_centers[:, np.newaxis] + z_rel[np.newaxis, :]
    z_pks = z_pks_mat.ravel(order="C")

    z_P1 = +(width_cicc + gap_DP_int) / 2.0
    z_P2 = -(width_cicc + gap_DP_int) / 2.0
    R_DP = np.concatenate([r_layers[::-1], r_layers])
    Z_DP = np.concatenate([np.full(num_turns, z_P1), np.full(num_turns, z_P2)])

    # =================================================================
    #   HIGH-ACCURACY 3D BIOT-SAVART via segmentMagneticField1d (prof.)
    # =================================================================
    M_seg_B = 32
    theta_pB = np.arange(M_seg_B, dtype=float) * (2.0 * np.pi / M_seg_B)

    # --- Sorgenti: TUTTE le spire (num_turns x N_pancakes) come poligoni ---
    N_loops_src = num_turns * N_pancakes
    N_edges_src = N_loops_src * M_seg_B

    k_idx_grid, p_idx_grid = np.meshgrid(
        np.arange(num_turns), np.arange(N_pancakes), indexing="ij"
    )
    k_idx = k_idx_grid.ravel(order="F")
    p_idx = p_idx_grid.ravel(order="F")

    R_loops = r_layers[k_idx]
    Z_loops = z_pks[p_idx]

    loop_id_node = np.repeat(np.arange(N_loops_src), M_seg_B)
    th_node = np.tile(theta_pB, N_loops_src)

    P_src = np.column_stack(
        [
            R_loops[loop_id_node] * np.cos(th_node),
            R_loops[loop_id_node] * np.sin(th_node),
            Z_loops[loop_id_node],
        ]
    )

    s_base = np.arange(M_seg_B)
    sn_base = (s_base + 1) % M_seg_B
    loop_offsets = np.repeat(np.arange(N_loops_src) * M_seg_B, M_seg_B)
    E_src = np.column_stack(
        [loop_offsets + np.tile(s_base, N_loops_src), loop_offsets + np.tile(sn_base, N_loops_src)]
    )

    # --- Punti di osservazione: 6 slot x N_sp_dp spire del DP equatoriale ---
    SpG, SlG = np.meshgrid(np.arange(N_sp_dp), np.arange(n_channels), indexing="ij")
    th_eval_slot = theta_slots[SlG.ravel(order="F")]
    R_obs = R_DP[SpG.ravel(order="F")] + R_slot * np.cos(th_eval_slot)
    Z_obs = Z_DP[SpG.ravel(order="F")] + R_slot * np.sin(th_eval_slot)
    phi_azim = np.zeros(R_obs.size, dtype=float)
    Q = np.column_stack(
        [R_obs * np.cos(phi_azim), R_obs * np.sin(phi_azim), Z_obs]
    )

    cprintf(
        True,
        "  segmentMagneticField1d: %d sorgenti, %d obs points ...\n",
        N_edges_src,
        Q.shape[0],
    )
    t_BS = time.perf_counter()
    H = segmentMagneticField1d(P_src, E_src, I_op, Q)
    cprintf(True, "  done in %.1f s\n", time.perf_counter() - t_BS)

    # Conversione H -> B (mu0 * H) e proiezione Cartesian -> cilindrico
    B_cart = mu0 * H
    Bx = B_cart[:, 0]
    By = B_cart[:, 1]
    Bz_pt = B_cart[:, 2]
    Br_pt = Bx * np.cos(phi_azim) + By * np.sin(phi_azim)

    # Reshape per spira x slot (Fortran order come in MATLAB reshape)
    Br_slots = np.reshape(Br_pt, (N_sp_dp, n_channels), order="F")
    Bz_slots = np.reshape(Bz_pt, (N_sp_dp, n_channels), order="F")

    # Aggregazione MAX sui 6 slot (caso peggiore)
    B_perp_per_turn = np.max(np.abs(Bz_slots), axis=1)
    B_par_per_turn = np.max(np.abs(Br_slots), axis=1)
    B_mod_per_turn = np.max(np.sqrt(Br_slots ** 2 + Bz_slots ** 2), axis=1)
    gamma_aniso = tape.gamma_aniso
    B_eff_per_turn = np.sqrt(B_perp_per_turn ** 2 + (gamma_aniso * B_par_per_turn) ** 2)

    # Diagnostica: confronto rapido con auto-tuning
    B_max_3D = float(np.max(B_mod_per_turn))
    cprintf(True, "  B_max (auto-tuning 2D) : %.3f T\n", B_max_val)
    cprintf(
        True,
        "  B_max (3D segment fn)  : %.3f T  (delta %.2f%%)\n",
        B_max_3D,
        100.0 * (B_max_3D - B_max_val) / B_max_val,
    )

    # Ic locale anisotropo (Kim)
    Ic_tape_per_turn = 150.0
    I_per_tape = I_op / geom.N_tapes
    load_frac = np.full(N_sp_dp, I_per_tape / Ic_tape_per_turn, dtype=float)
    wi = int(np.argmax(load_frac)) + 1  # 1-based index for display consistency with MATLAB
    cprintf(
        True,
        "  Ic range  : [%.1f, %.1f] A/tape (I_per_tape = %.1f A)\n",
        np.min(Ic_tape_per_turn),
        np.max(Ic_tape_per_turn),
        I_per_tape,
    )
    cprintf(
        True,
        "  Load_frac : [%.3f, %.3f] (worst spira: %d)\n",
        np.min(load_frac),
        np.max(load_frac),
        wi,
    )

    # L_dp via inductanceMatrix1d (polygon approximation)
    M_seg = 8
    theta_poly = np.arange(M_seg, dtype=float) * (2.0 * np.pi / M_seg)
    ii_loop = np.repeat(np.arange(N_sp_dp), M_seg)
    th_vec = np.tile(theta_poly, N_sp_dp)
    P = np.column_stack(
        [
            R_DP[ii_loop] * np.cos(th_vec),
            R_DP[ii_loop] * np.sin(th_vec),
            Z_DP[ii_loop],
        ]
    )
    N_edges_E = N_sp_dp * M_seg
    s_seg = np.arange(M_seg)
    sn_seg = (s_seg + 1) % M_seg
    seg_offsets = np.repeat(np.arange(N_sp_dp) * M_seg, M_seg)
    E = np.column_stack(
        [seg_offsets + np.tile(s_seg, N_sp_dp), seg_offsets + np.tile(sn_seg, N_sp_dp)]
    )
    L_member = sp.csr_matrix(
        (np.ones(N_edges_E, dtype=float), (np.arange(N_edges_E), ii_loop)),
        shape=(N_edges_E, N_sp_dp),
    )

    cprintf(
        True,
        "  Calcolo L_dp via inductanceMatrix1d (M_seg=%d, %d edges)...\n",
        M_seg,
        N_edges_E,
    )
    t_L = time.perf_counter()
    M_seg_mat = inductanceMatrix1d(P, E, a_wire)
    cprintf(True, "  done in %.1f s\n", time.perf_counter() - t_L)
    L_dp = (L_member.T @ M_seg_mat @ L_member)
    if sp.issparse(L_dp):
        L_dp = L_dp.toarray()
    L_total_dp = float(np.sum(L_dp))
    cprintf(
        True,
        "  L_self_media = %.2f uH/spira, L_DP_total = %.2f mH\n",
        np.mean(np.diag(L_dp)) * 1e6,
        L_total_dp * 1e3,
    )

    # V_ext per spira (mutue verso pancake non-DP) - precomputed scalar per dI/dt
    ic = N_pancakes // 2  # 1-based in MATLAB: ic and ic+1 -> 0-based: ic-1 and ic
    idx_pks_ext = np.setdiff1d(np.arange(N_pancakes), np.array([ic - 1, ic]))
    M_ext_per_turn = np.zeros(N_sp_dp, dtype=float)
    if idx_pks_ext.size > 0:
        Ki_ext, Pi_ext = np.meshgrid(
            np.arange(num_turns), idx_pks_ext, indexing="ij"
        )
        R_ev = r_layers[Ki_ext.ravel(order="F")]
        Z_ev = z_pks[Pi_ext.ravel(order="F")]
        R1m = R_DP[:, np.newaxis]
        R2m = R_ev[np.newaxis, :]
        Z1m = Z_DP[:, np.newaxis]
        Z2m = Z_ev[np.newaxis, :]
        dm = np.abs(Z1m - Z2m)
        k2m = 4.0 * R1m * R2m / ((R1m + R2m) ** 2 + dm ** 2)
        k2m = np.clip(k2m, 1e-14, 1.0 - 1e-14)
        km = np.sqrt(k2m)
        K_ell = ellipk(k2m)
        E_ell = ellipe(k2m)
        M_ext_mat = (
            mu0
            * np.sqrt(R1m * R2m)
            * ((2.0 / km - km) * K_ell - (2.0 / km) * E_ell)
        )
        M_ext_per_turn = np.sum(M_ext_mat, axis=1)

    # =====================================================================
    # Calcolo Baseline della Resistenza (per i grafici)
    # =====================================================================
    E_c = 1e-4
    n_pl = 25
    A_tape = p.w_tape * p.thick_tape
    A_REBCO_total = geom.N_tapes * A_tape

    # Densità di corrente operativa e critica
    J_op = I_op / A_REBCO_total
    J_c_per_turn = Ic_tape_per_turn / A_tape

    # Resistività Power-law locale
    rho_pl = (E_c / J_c_per_turn) * ((J_op / J_c_per_turn) ** (n_pl - 1))

    # Lunghezza di ogni singola spira e Resistenza totale per spira
    L_per_turn = 2.0 * np.pi * R_DP
    R_per_turn_b = rho_pl * L_per_turn / A_REBCO_total

    # =====================================================================
    # Plots
    # =====================================================================
    turn_idx = np.arange(1, N_sp_dp + 1)

    # Fig 1 - EM: B_perp, B_par, B_mod vs turn
    plt.figure(num="01 - Baseline EM: B vs spira", figsize=(9.0, 4.2))
    plt.plot(turn_idx, B_perp_per_turn, "r-s", linewidth=2, label=r"$B_{perp}$")
    plt.plot(turn_idx, B_par_per_turn, "b-o", linewidth=2, label=r"$B_{par}$")
    plt.plot(turn_idx, B_mod_per_turn, "k-^", linewidth=1.5, label=r"$|B|$")
    plt.plot(turn_idx, B_eff_per_turn, "m--", linewidth=1.5, label=r"$B_{eff}$ (aniso)")
    plt.axvline(num_turns + 0.5, color="g", linestyle="--", label="P1|P2")
    plt.grid(True)
    plt.xlabel("Indice spira")
    plt.ylabel("B [T]")
    plt.legend(loc="best")
    plt.title(f"Campo magnetico per spira ($B_{{max}}$={B_max_val:.2f} T)")
    plt.tight_layout()

    # Fig 2 - EM: Resistance vs turn (log)
    plt.figure(num="02 - Baseline EM: R per spira (log)", figsize=(9.0, 4.2))
    plt.semilogy(turn_idx, R_per_turn_b, "b-o", linewidth=2)
    plt.axvline(num_turns + 0.5, color="g", linestyle="--", label="P1|P2")
    plt.grid(True)
    plt.xlabel("Indice spira")
    plt.ylabel(r"R per spira [$\Omega$]")
    plt.title(f"Resistenza elettrica per spira @ T={T_baseline}K (scala log)")
    plt.tight_layout()

    # Fig 3 - EM: Load fraction vs turn
    plt.figure(num="03 - Baseline EM: Load fraction", figsize=(9.0, 4.2))
    plt.plot(turn_idx, load_frac, "b-o", linewidth=2)
    plt.axhline(
        p.f_margin,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"$f_{{margin}}$={p.f_margin:.2f}",
    )
    plt.plot(wi, load_frac[wi - 1], "r*", markersize=16, linewidth=2)
    plt.axvline(num_turns + 0.5, color="g", linestyle="--", label="P1|P2")
    plt.grid(True)
    plt.xlabel("Indice spira")
    plt.ylabel(r"$I_{op}/(N_{tapes}\cdot I_c)$")
    plt.title("Frazione di carico Ic locale")
    plt.tight_layout()

    # Fig 4 - Geometry: CICC cross-section + axial pancake stack
    fig4, (ax1, ax2) = plt.subplots(
        1, 2, num="04 - Baseline Geom", figsize=(11.0, 4.6)
    )
    ax1.set_aspect("equal")
    theta_pl = np.arange(n_channels, dtype=float) * (2.0 * np.pi / n_channels)
    z_P1_plot = +width_cicc / 2.0
    r_layers_plot = R_in + np.arange(num_turns, dtype=float) * width_cicc + width_cicc / 2.0

    for i in range(num_turns):
        r_cicc = r_layers_plot[i]
        # 1) Jacket SS
        ax1.add_patch(
            Rectangle(
                (r_cicc - width_cicc / 2.0, z_P1_plot - width_cicc / 2.0),
                width_cicc,
                width_cicc,
                edgecolor="k",
                linewidth=1.2,
                facecolor=(0.95, 0.95, 0.95),
            )
        )
        # 2) Copper Former
        ax1.add_patch(
            Circle(
                (r_cicc, z_P1_plot),
                geom.R_former,
                edgecolor="#b87333",
                facecolor="#f5d4ad",
                linewidth=1.2,
            )
        )
        # 3) He Hole
        ax1.add_patch(
            Circle(
                (r_cicc, z_P1_plot),
                geom.R_central,
                edgecolor="b",
                facecolor="c",
                linewidth=1.0,
            )
        )
        # 4) REBCO Slots
        for ch in range(n_channels):
            rcx = r_cicc + geom.R_slot * np.cos(theta_pl[ch])
            rcy = z_P1_plot + geom.R_slot * np.sin(theta_pl[ch])
            ax1.add_patch(
                Rectangle(
                    (rcx - geom.w_slot / 2.0, rcy - geom.h_slot / 2.0),
                    geom.w_slot,
                    geom.h_slot,
                    edgecolor="k",
                    facecolor="r",
                )
            )

    ax1.grid(True)
    ax1.set_xlabel("Raggio del solenoide [m]")
    ax1.set_ylabel("Quota Z [m]")
    ax1.set_title("Sezione CICC del Pancake")
    ax1.set_xlim(
        R_in - 0.5 * width_cicc, R_in + num_turns * width_cicc + 0.5 * width_cicc
    )
    ax1.set_ylim(z_P1_plot - width_cicc, z_P1_plot + width_cicc)

    ax2.plot(
        np.zeros(N_pancakes),
        z_pks,
        "o",
        markersize=9,
        markerfacecolor=(0.7, 0.7, 0.7),
        markeredgecolor=(0.4, 0.4, 0.4),
    )
    ax2.plot(
        np.zeros(2),
        z_pks[ic - 1 : ic + 1],
        "o",
        markersize=11,
        markerfacecolor="#4169E1",
        markeredgecolor="k",
    )
    ax2.axhline(0.0, color="r", linestyle="--", label="Piano equatoriale")
    ax2.axhline(+H_sol_max / 2.0, color="k", linestyle=":")
    ax2.axhline(-H_sol_max / 2.0, color="k", linestyle=":")
    ax2.grid(True)
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-H_sol_max / 2.0 * 1.05, H_sol_max / 2.0 * 1.05)
    ax2.set_xticks([])
    ax2.set_ylabel("Quota Z [m]")
    ax2.set_title(f"Stack: {N_pancakes} pancake")
    plt.tight_layout()

    # %% =====================================================================
    #  SALVATAGGIO DATI PER LO STEP 6 (TRANSIENTE)
    #  =====================================================================
    cprintf(True, "\nSalvataggio dati in MagneticData_Final_nuovissima.mat...\n")

    L_per_turn = 2.0 * np.pi * R_DP
    xi_centroid = np.cumsum(L_per_turn) - L_per_turn / 2.0
    L_cable_DP = float(np.sum(L_per_turn))

    mag_dict = {
        "N_sp_dp": N_sp_dp,
        "num_turns": num_turns,
        "N_pancakes": N_pancakes,
        "H_active": H_active,
        "R_DP": R_DP,
        "Z_DP": Z_DP,
        "L_per_turn": L_per_turn,
        "xi_centroid": xi_centroid,
        "L_cable_DP": L_cable_DP,
        "B_perp_per_turn": B_perp_per_turn,
        "B_par_per_turn": B_par_per_turn,
        "B_mod_per_turn": B_mod_per_turn,
        "B_eff_per_turn": B_eff_per_turn,
        "L_dp": L_dp,
        "M_ext_per_turn": M_ext_per_turn,
    }
    mat_out = os.path.join(script_dir, "MagneticData_Final_nuovissima.mat")
    sio.savemat(mat_out, mag_dict)

    cprintf(True, "Fatto! Dati magnetici pronti per lo Step 6.\n")

    if show_plots:
        plt.show()

    return ProjectStruct(**mag_dict)


if __name__ == "__main__":
    run_step2_magnetic_design(show_plots=True)
