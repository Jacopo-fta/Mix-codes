# =========================================================================
# Step 3 - CICC Layout & Dimensioning (VERSIONE _nuovissima)
#  Stratificazione reale del nastro (Civale/Tamagnini) + tau_coupling fisico.
#
#  Input : setupProject_nuovissima.py
#  Output: CICC_Geometry_nuovissima.mat (consumato da Step2_..._nuovissima)
# =========================================================================
import os
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle, Patch
from setupProject_nuovissima import setupProject_nuovissima, ProjectStruct


def run_dimensionamento_CICC(show_plots=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = setupProject_nuovissima()

    # %% 1. Parametri del modello di letteratura (REBCO)
    T_c = 92.0          # [K] temperatura critica
    s_T = 2.0           # [-] esponente scaling termico
    B0_op = 1.0         # [T] parametro B0 del modello Kim
    alpha_op = 0.75     # [-] esponente alpha
    beta_op = 1.0       # [-] esponente beta
    gamma_ani = 5.2     # [-] fattore di anisotropia

    T_op = 20.0         # [K] temperatura operativa richiesta

    # Corrente critica a campo nullo per un singolo nastro (4mm x spessore)
    A_tape_single = p.w_tape * p.thick_tape       # ingombro nastro (100um) - per geometria
    A_SC_single = p.w_tape * p.thick_sc_tape      # SOLO film ReBCO (1um) - per Jc/perdite
    Ic_0T_op = 120.0 * 11.7
    J_c0_op = Ic_0T_op / A_SC_single

    # %% 2. Scaling Termico: Jc(0T, T_op)
    F_T = ((T_c - T_op) / T_c) ** s_T

    # J_c0_base = Jc del SOLO ReBCO (riferito all'area SC reale 1um)
    J_c0_base = Ic_0T_op / (A_SC_single * F_T)

    # %% 3. Calcolo Ic al punto di design (SIZING AUTO-CONSISTENTE con Kim)
    B_eff_design = p.B_target * p.aniso_design_factor
    Ic_op = 120.0 * 1.3  # Ic(B_design, T_op) da Kim

    print("======== Proprietà del Nastro (Modello Anisotropo) ========")
    print("Parametri Modello:")
    print(f"  Jc0 (0K, 0T)      : {J_c0_base:.2e} A/m^2")
    print(f"  B0                : {B0_op:.3f} T")
    print(f"  Alpha             : {alpha_op:.2f}")
    print(f"  Gamma (aniso)     : {gamma_ani:.2f}")
    print("Condizioni Operative:")
    print(f"  T_op              : {int(T_op)} K")
    print(f"  F_T (0K -> {int(T_op)}K)   : {F_T:.4f}")
    print(f"  Ic(0T, {int(T_op)}K)       : {Ic_0T_op:.2f} A")
    print(f"  Ic({B_eff_design:.1f} T, {int(T_op)}K)   : {Ic_op:.2f} A (caso peggiore theta)")
    print("===========================================================\n")

    # %% 4. Numero di nastri (Topologia a 6 slot rettangolari)
    n_channels = 6
    N_tapes_min = math.ceil(p.I_op / (Ic_op * p.f_load_design))
    tapes_per_slot = math.ceil(N_tapes_min / n_channels)
    N_tapes = tapes_per_slot * n_channels

    # --- Diagnostica robustezza: T_cs stimata al punto di design -------------
    I_per_tape_des = p.I_op / N_tapes
    load_frac_des = I_per_tape_des / Ic_op
    T_cs_des = T_c - (T_c - T_op) * (load_frac_des ** (1.0 / s_T))
    print(
        f"  [Sizing robusto] B_design={B_eff_design:.1f} T, Ic_op={Ic_op:.1f} A, f_load={p.f_load_design:.2f}"
    )
    print(
        f"  [Sizing robusto] N_tapes={N_tapes}, I/tape={I_per_tape_des:.1f} A, load={load_frac_des:.2f} -> T_cs_design={T_cs_des:.1f} K (headroom {T_cs_des - T_op - p.dT_margin:.1f} K)"
    )

    A_sc = N_tapes * A_tape_single  # Area ingombro nastri (bundle, 100um)

    # --- Aree per materiale (stratificazione reale del nastro) ---------------
    A_SC_real = N_tapes * p.w_tape * p.thick_sc_tape   # ReBCO (1um)
    A_Cu_tape = N_tapes * p.w_tape * p.thick_cu_tape   # Cu nel nastro (40um)
    A_Hast = N_tapes * p.w_tape * p.thick_hast         # substrato Hastelloy (50um)
    A_Ag = N_tapes * p.w_tape * p.thick_ag             # overlayer Ag (2um)

    # %% 5. Copper former e Aree Elio
    Cu_SC_ratio = 5.0
    A_cu = A_sc * Cu_SC_ratio

    void_fraction_global = 0.35
    A_flow_total = (A_sc + A_cu) * (void_fraction_global / (1.0 - void_fraction_global))

    # %% 6. Definizione Slot Rettangolari (6 slot)
    w_slot = p.w_tape  # Larghezza slot = larghezza nastro (4mm)
    h_stack = tapes_per_slot * p.thick_tape
    void_fraction_slot = 0.20
    h_slot = h_stack / (1.0 - void_fraction_slot)

    A_slot_gross = w_slot * h_slot
    A_he_in_slots = n_channels * (A_slot_gross - (p.w_tape * h_stack))

    # Raggio equivalente per induttanza (spira)
    a_wire = math.sqrt(A_slot_gross / math.pi)

    # %% 7. Foro centrale e Geometria Finale
    A_central_he = A_flow_total - A_he_in_slots
    if A_central_he < 0:
        A_central_he = 0.0
    R_central = math.sqrt(A_central_he / math.pi)

    A_former_gross = A_cu + A_central_he + n_channels * A_slot_gross
    R_former = math.sqrt(A_former_gross / math.pi)

    R_slot = R_central + (R_former - R_central) / 2.0

    # Jacket SS
    thickness_ss = 2e-3
    side_external = (2.0 * R_former) + 2.0 * thickness_ss

    # %% 7b. Coupling time-constant inter-tape con TWIST (Lp imposto dal docente)
    t_contact = p.thick_tape                     # pitch SC-SC nello stack [m]
    rho_t_eff = p.Rs_contact / t_contact         # [Ohm*m]
    L_loop_twist = p.L_twist / (2.0 * math.pi)   # [m] scala geometrica del loop twistato
    tau_coupling = p.mu0 * (L_loop_twist ** 2) / (p.k_geom_twist * rho_t_eff)
    print(f"  twist             : Lp = {p.L_twist*1e3:.0f} mm -> L_loop = Lp/(2pi) = {L_loop_twist*1e3:.2f} mm")
    print(f"  tau_coupling      : {tau_coupling:.3e} s (twisted, da Rs={p.Rs_contact:.2e} Ohm*m^2)")

    # %% 8. Salvataggio e Output
    print("======== Layout CICC (6 Slot) ========")
    print(f"Nastri totali      : {N_tapes} ({tapes_per_slot} per slot)")
    print(f"Dimensioni Slot    : {w_slot*1e3:.2f} x {h_slot*1e3:.2f} mm")
    print(f"Raggio Former      : {R_former*1e3:.2f} mm")
    print(f"Lato Esterno CICC  : {side_external*1e3:.2f} mm")
    print("======================================")

    geom = ProjectStruct(
        N_tapes=N_tapes,
        tapes_per_slot=tapes_per_slot,
        n_channels=n_channels,
        w_slot=w_slot,
        h_slot=h_slot,
        R_slot=R_slot,
        R_central=R_central,
        R_former=R_former,
        a_wire=a_wire,
        width_cicc=side_external,
        thickness_ss=thickness_ss,
    )

    tape = ProjectStruct(
        Ic_0T_op=Ic_0T_op,
        B0_op=B0_op,
        alpha_op=alpha_op,
        beta_op=beta_op,
        gamma_aniso=gamma_ani,
        Tc=T_c,
        s_T=s_T,
        Ic_op_design=Ic_op,
        J_c0_base=J_c0_base,
        A_SC_real=A_SC_real,
        A_Cu_tape=A_Cu_tape,
        A_Hast=A_Hast,
        A_Ag=A_Ag,
        tau_coupling=tau_coupling,
        L_twist=p.L_twist,
    )

    mat_path = os.path.join(script_dir, "CICC_Geometry_nuovissima.mat")
    sio.savemat(mat_path, {"geom": geom.to_dict(), "tape": tape.to_dict()})

    # %% 9. Plot Sezione Trasversale del Singolo CICC
    fig, ax = plt.subplots(
        num="Sezione Trasversale Singolo CICC", figsize=(6.5, 6.5), facecolor="w"
    )
    ax.set_aspect("equal")
    ax.grid(True)

    # Palette Colori
    col_ss = (0.80, 0.82, 0.85)  # Jacket Acciaio Inox (grigio chiaro)
    col_cu = (0.85, 0.55, 0.30)  # Former Rame (rame/bronzo)
    col_he = (0.45, 0.80, 1.00)  # Canali Elio (azzurro)
    col_sc = (0.15, 0.15, 0.15)  # Stack Nastri REBCO (antracite scuro)

    # 1. Disegno Jacket Esterno Quadrato (incluso spessore)
    L_ext = side_external * 1e3  # [mm]
    ax.add_patch(
        Rectangle(
            (-L_ext / 2.0, -L_ext / 2.0),
            L_ext,
            L_ext,
            facecolor=col_ss,
            edgecolor="k",
            linewidth=1.5,
        )
    )

    # 2. Former Circolare in Rame
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            R_former * 1e3,
            facecolor=col_cu,
            edgecolor="k",
            linewidth=1.2,
        )
    )

    # 3. I 6 Slot Rettangolari con Stack REBCO ed Elio
    angles_deg = np.arange(0, 360, 60)  # 6 canali a 60 gradi
    w_s_mm = w_slot * 1e3
    h_s_mm = h_slot * 1e3
    h_st_mm = h_stack * 1e3
    r_mid_mm = R_slot * 1e3

    for k in range(n_channels):
        ang = np.deg2rad(angles_deg[k])
        R_rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

        # Contorno Slot (Elio bagnato)
        p_slot_local = np.array(
            [
                [-w_s_mm / 2.0, -h_s_mm / 2.0],
                [w_s_mm / 2.0, -h_s_mm / 2.0],
                [w_s_mm / 2.0, h_s_mm / 2.0],
                [-w_s_mm / 2.0, h_s_mm / 2.0],
            ]
        ).T
        p_slot_local[0, :] += r_mid_mm
        p_slot_global = (R_rot @ p_slot_local).T
        ax.add_patch(
            Polygon(
                p_slot_global,
                closed=True,
                facecolor=col_he,
                edgecolor=(0.0, 0.4, 0.8),
                linewidth=1.0,
            )
        )

        # Stack REBCO all'interno dello slot
        p_sc_local = np.array(
            [
                [-w_s_mm / 2.0, -h_st_mm / 2.0],
                [w_s_mm / 2.0, -h_st_mm / 2.0],
                [w_s_mm / 2.0, h_st_mm / 2.0],
                [-w_s_mm / 2.0, h_st_mm / 2.0],
            ]
        ).T
        p_sc_local[0, :] += r_mid_mm
        p_sc_global = (R_rot @ p_sc_local).T
        ax.add_patch(
            Polygon(
                p_sc_global,
                closed=True,
                facecolor=col_sc,
                edgecolor="k",
                linewidth=0.8,
            )
        )

    # 4. Canale Centrale di Flusso dell'Elio
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            R_central * 1e3,
            facecolor=col_he,
            edgecolor=(0.0, 0.4, 0.8),
            linewidth=1.5,
        )
    )

    # Limiti e Decorazioni Grafiche
    lim_plot = (L_ext / 2.0) * 1.12
    ax.set_xlim(-lim_plot, lim_plot)
    ax.set_ylim(-lim_plot, lim_plot)
    ax.set_xlabel("X [mm]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Y [mm]", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Sezione Singolo CICC (6 Slot, {N_tapes} Nastri REBCO)",
        fontsize=12,
        fontweight="bold",
    )

    # Legenda Componenti
    h_leg = [
        Patch(facecolor=col_ss, edgecolor="k", label="Jacket AISI 316LN"),
        Patch(facecolor=col_cu, edgecolor="k", label="Former in Rame (Cu)"),
        Patch(facecolor=col_sc, edgecolor="k", label="Stack REBCO (HTS)"),
        Patch(facecolor=col_he, edgecolor=(0.0, 0.4, 0.8), label="Canali Flusso Elio (He)"),
    ]
    ax.legend(
        handles=h_leg,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        fontsize=9,
    )
    plt.tight_layout()

    if show_plots:
        plt.show()

    return geom, tape


if __name__ == "__main__":
    run_dimensionamento_CICC(show_plots=True)
