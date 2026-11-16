# =========================================================================
# main.py - Esecuzione completa in sequenza del progetto Superconducting Magnets
#  1) dimensionamento_CICC_nuovissima.py  -> genera CICC_Geometry_nuovissima.mat
#  2) Step2_MagneticDesign_nuovissima.py  -> genera MagneticData_Final_nuovissima.mat
#  3) Step6_optimusPrime_nuovissima_c.py  -> simulazione EM<->TH 2-way e bisezione mdot
# =========================================================================
import sys
import matplotlib.pyplot as plt

from dimensionamento_CICC_nuovissima import run_dimensionamento_CICC
from Step2_MagneticDesign_nuovissima import run_step2_magnetic_design
from Step6_optimusPrime_nuovissima_c import run_step6_optimus_prime


def main(show_plots=True):
    print("=========================================================================")
    print("  STEP 1/3 : Dimensionamento CICC (dimensionamento_CICC_nuovissima.py)")
    print("=========================================================================")
    run_dimensionamento_CICC(show_plots=False)

    print("\n=========================================================================")
    print("  STEP 2/3 : Magnetic Design (Step2_MagneticDesign_nuovissima.py)")
    print("=========================================================================")
    run_step2_magnetic_design(show_plots=False)

    print("\n=========================================================================")
    print("  STEP 3/3 : Simulazione EM-TH 2-Way (Step6_optimusPrime_nuovissima_c.py)")
    print("=========================================================================")
    run_step6_optimus_prime(show_plots=False)

    if show_plots:
        plt.show()


if __name__ == "__main__":
    no_plot = "--no-plot" in sys.argv
    main(show_plots=not no_plot)
