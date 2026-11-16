from types import SimpleNamespace
import numpy as np


class ProjectStruct(SimpleNamespace):
    """Dictionary- and attribute-accessible struct (MATLAB struct equivalent)."""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        return self.__dict__.keys()

    def to_dict(self):
        return dict(self.__dict__)


def setupProject_nuovissima():
    """
    SETUPPROJECT_NUOVISSIMA  Parametri globali - versione con stratificazione
    reale del nastro (Civale/Tamagnini) e parametri di resistenza fisici.

    USO:
        p = setupProject_nuovissima()
        p.I_op, p.B_target, p.R_in, ...
    """
    p = ProjectStruct()

    # --- Specifiche di progetto (traccia Prof.) -----------------------------
    p.I_op = 50000.0        # Corrente operativa [A]
    p.B_target = 15.0       # Campo MASSIMO sul conduttore [T]
    p.R_in = 0.50           # Raggio interno solenoide [m]
    p.H_sol_max = 1.50      # Altezza massima del solenoide [m]
    p.dIdt = 200.0          # Ramp rate [A/s] (= 0.2 kA/s)
    p.dT_margin = 5.0       # Temperature margin richiesto [K]

    # --- Materiali ----------------------------------------------------------
    p.RRR_cu = 100.0        # Stabilizzatore Cu, RRR
    p.Tc_rebco = 92.0       # Tc REBCO [K]
    p.scaling_s = 2.0       # Esponente di (1-T/Tc)^s per Ic vs T

    # --- Refrigerante (elio) ------------------------------------------------
    p.mdot_max = 15e-3      # [kg/s]
    p.T_in_min = 4.5        # [K]
    p.T_in_max = 50.0       # [K]
    p.p_in_min = 5e5        # [Pa]
    p.p_in_max = 20e5       # [Pa]

    # --- Geometria nastro (stratificazione reale - Civale/Tamagnini) --------
    p.w_tape = 4e-3             # [m] larghezza nastro reale
    p.thick_sc_tape = 1e-6      # [m] ReBCO (HTS) - Civale ~1um
    p.thick_cu_tape = 40e-6     # [m] Cu stabilizzatore nel nastro (20 top + 20 bottom)
    p.thick_hast = 50e-6        # [m] substrato Hastelloy
    p.thick_ag = 2e-6           # [m] overlayer Ag
    p.thick_tape = 0.1e-3       # [m] spessore d'ingombro (~7um = buffer/tolleranza)
    p.w_bridge_lab = 0.3e-3     # [m] larghezza ponte laser usato in lab
    p.n_filaments = 20          # [-] filamenti per striazione del nastro

    # --- Resistenze di accoppiamento ----------------------------------------
    p.r_contact_JS2 = 1e-4      # [K*m^2/W] resistenza di contatto TERMICA conduttore-jacket
    p.Rs_contact = 1.5e-9       # [Ohm*m^2] resistenza di contatto ELETTRICA inter-tape

    # --- Twist del cavo (richiesta esplicita del docente) -------------------
    p.L_twist = 150e-3          # [m] passo di twist del cavo CICC. Tipico 50-150 mm.
    p.k_geom_twist = 2.0        # [-] fattore geometrico Wilson per cavo twistato

    # --- Contesto della misura di laboratorio -------------------------------
    p.T_lab = 77.0              # [K] LN2
    p.T_op = 20.0               # [K] temperatura di esercizio scelta

    p.B_lab_no_magnet = 0.0     # [T]
    p.B_lab_with_magnet = 5.0   # [T]

    # --- Parametri operativi ------------------------------------------------
    p.f_margin = 0.85           # Frazione di carico Ic (Iop/Ic <= f_margin) [legacy]

    # --- Sizing ROBUSTO auto-consistente con Kim (vedi dimensionamento) ------
    p.f_load_design = 0.93      # [-] frazione di carico target al punto di design
    p.aniso_design_factor = 1.05

    # --- Costanti fisiche ---------------------------------------------------
    p.mu0 = 4.0 * np.pi * 1e-7

    return p


if __name__ == "__main__":
    p = setupProject_nuovissima()
    print("Parametri progetto caricati con successo:")
    for k, v in p.to_dict().items():
        print(f"  {k:20s} = {v}")
