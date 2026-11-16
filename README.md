# Nuclar Engineering MSc Projects

Collection of Python scripts and numerical models developed for Nuclear Engineering MSc coursework, covering reactor thermal-hydraulics, fuel pin mechanics, radiation protection geometry, and superconducting magnet design.

## Repository Structure

### `Fission_plants/`
Thermal-hydraulic and structural analyses for pressurized water reactors (AP1000):
- `Assignment1.py` / `Assignment1_2.py`: Two-phase natural circulation and coupled multi-loop passive decay heat removal systems.
- `Assignment2.py`: 1D axial steady-state thermal-hydraulic analysis of the AP1000 average and hot channels (boiling onset, void fraction, radial fuel temperature profile, W-3 CHF, MDNBR). Outputs `Tci_data.mat`, `Tco_data.mat`, and `p_in_data.mat`.
- `Assignment3.py`: End-of-life Zircaloy-4 cladding thermo-mechanical verification (ASME Section III, Tresca criterion), fission gas release inventory, and gas plenum sizing using the outputs of `Assignment2.py`.

### `Radiation_Protection/`
- `Geometry.py` / `main.py`: Geometric and uranium mass calculations for a VVER-440 fuel assembly (IAEA TRS 379).

### `SC_Project/`
Design and coupled electromagnetic/thermal-hydraulic simulation of a 15 T REBCO superconducting solenoid using a 6-slot CICC conductor:
- `setupProject_nuovissima.py`: Global design specifications, tape stratification, and material/contact parameters.
- `dimensionamento_CICC_nuovissima.py`: CICC cross-section layout (6 slots, copper former, helium channels, SS jacket) and twist coupling time constant. Saves `CICC_Geometry_nuovissima.mat`.
- `Step2_MagneticDesign_nuovissima.py`: Solenoid pancake/turn optimization, 3D Biot-Savart field evaluation (`segmentMagneticField1d`), and double-pancake inductance matrix (`inductanceMatrix1d`). Saves `MagneticData_Final_nuovissima.mat`.
- `Step6_optimusPrime_nuovissima_c.py`: 2-way coupled EM-TH transient solver (3-equation BDF1 + upwind helium flow, dynamic AC losses, `directSolve` circuit network) with bisection for the minimum helium mass flow rate.
- `main.py`: Runs the three design and simulation steps in sequence (supports `--no-plot` to skip figure windows).
- Supporting electromagnetic and linear solver modules: `Mu0.py`, `biot_savart_loops_vec.py`, `segmentMagneticField1d.py`, `inductanceMatrix1d.py`, `mutual_inductance_maxwell_grover.py`, `estrai_Ic.py`, `directSolve.py`, `factorizeMatrix.py`, `forwardBackwardSolve.py`, `checkMatrixStructure.py`, `assignFixValue.py`, `assembleSolution.py`, `iterativeRefinement.py`, `cond1est.py`, `cprintf.py`.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

Main packages used across the scripts: `numpy`, `scipy`, `matplotlib`, `pyXSteam`, `CoolProp`, `pandas`, `pyyaml`.

## Usage

Run any script from the repository root.

Fission plants (run `Assignment2.py` before `Assignment3.py`):
```bash
python Fission_plants/Assignment1.py
python Fission_plants/Assignment1_2.py
python Fission_plants/Assignment2.py
python Fission_plants/Assignment3.py
```

Radiation protection:
```bash
python Radiation_Protection/Geometry.py
```

Superconducting magnet project (step by step or full pipeline):
```bash
python SC_Project/dimensionamento_CICC_nuovissima.py
python SC_Project/Step2_MagneticDesign_nuovissima.py
python SC_Project/Step6_optimusPrime_nuovissima_c.py
```
or:
```bash
python SC_Project/main.py
```
