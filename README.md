# ⚛️ Nuclear Engineering MSc — Thermal-Hydraulics & Fuel Rod Analysis

Welcome to this repository! This folder collects the computational assignments and numerical models developed during my **Master's Degree (Laurea Magistrale) in Nuclear Engineering**. 

Originally developed in MATLAB, these scripts have been fully ported to **Python** to create an open-source, reproducible, and easily accessible portfolio of nuclear reactor engineering analyses.

---

## 🎯 Purpose of the Repository (Obiettivo della Repository)

* **🇬🇧 English:** The main goal of this repository is to archive and showcase the practical modeling work carried out during my MSc in Nuclear Engineering. It covers core thermal-hydraulics, passive decay heat removal systems (natural circulation), two-phase flow heat transfer, Departure from Nucleate Boiling Ratio ($DNBR$) evaluation, and thermo-mechanical structural verification of a Pressurized Water Reactor (PWR - AP1000) fuel pin.
* **🇮🇹 Italiano:** Questa repository nasce con l'obiettivo di raccogliere, ordinare e condividere in Python i codici di calcolo sviluppati durante il percorso di **Laurea Magistrale in Ingegneria Nucleare**. I modelli affrontano problemi reali di progettazione termoidraulica e termomeccanica del nocciolo di reattori ad acqua pressurizzata (PWR - AP1000), spaziando dalla circolazione naturale passiva all'analisi del canale caldo e alla verifica strutturale del rivestimento secondo le normative ASME.

---

## 📂 Repository Structure & Assignments Overview

### 1. Natural Circulation & Passive Decay Heat Removal
* **`assignment1_1_c.py`**: Evaluates the required elevation head ($H_L$) to establish steady-state two-phase natural circulation in a loop with a heated section and a cooler, accounting for wall roughness ($\varepsilon$) and friction factors ($f_l, f_v$) via the Haaland correlation.
* **`assignment1_2.py`**: Models a coupled multi-loop passive decay heat removal system ($P_{decay} = 6\text{ MW}$, $1\%$ of nominal power) consisting of a Primary System Circuit ($PSC$), an Intermediate System Circuit ($ISC$), and two heat exchangers ($HX1$ and $HX2$) submerged in a boiling pool at $T_{sat} = 100\,^\circ\text{C}$. It iteratively solves the coupled thermal-hydraulic balance between buoyancy driving head ($\Delta P_{buoyancy}$) and distributed/localized pressure drops ($\Delta P_{loss}$).

### 2. AP1000 Core Thermal-Hydraulics & Hot Channel Analysis
* **`ass_2_finale.py`**: Performs a detailed 1D axial steady-state thermal-hydraulic analysis of an **AP1000** reactor core ($3400\text{ MW}_{th}$, $p_{nom} \approx 155.1\text{ bar}$):
  * Axial power distribution ($F_q = 2.60$), coolant enthalpy ($i$), and temperature profiles ($T_{cool}$) in both the average and hot channels.
  * Subcooled boiling onset (Jens-Lottes correlation), bubble detachment point (Griffith model), true flow quality ($x$), and void fraction ($\alpha$) axial profiles using **Zuber-Findlay**, **Fauske**, and **Moody** correlations.
  * Radial temperature distribution across the fuel pin: outer/inner cladding ($T_{co}, T_{ci}$), dynamic gap conductance ($h_{gap}$ via Ross & Stoute model accounting for thermal expansion and elastic deformation), fuel surface ($T_{fs}$), and fuel centerline temperature ($T_{fcl}$) via the Westinghouse $\text{UO}_2$ thermal conductivity integral.
  * Critical Heat Flux ($CHF$) evaluation using the **W-3 correlation**, grid spacer factor ($F_s$), **Tong non-uniform flux factor ($F$)**, and minimum Departure from Nucleate Boiling Ratio ($MDNBR$).
  * *Output:* Generates axial temperature plots and exports `Tci_data.mat`, `Tco_data.mat`, and `p_in_data.mat` for the structural analysis.

### 3. Fuel Pin Thermo-Mechanical Verification & Gas Plenum Sizing
* **`ass3_2.py`**: Performs the structural and thermo-mechanical verification of the **Zircaloy-4** cladding of an AP1000 fuel rod at End-Of-Life ($BU = 60\text{ MWd/kgU}$), using the temperature profiles generated in Assignment 2:
  * Preliminary elastic buckling verification ($p_{cr}$ vs. $p_{nom}$).
  * Maximum internal pressure limit ($p_{i,max}$) via Mariotte's formula.
  * Evaluation of primary mechanical stresses (hoop $\sigma_h$, radial $\sigma_r$, axial $\sigma_l$) and secondary thermal stresses across the cladding wall along the axial coordinate $z$.
  * **ASME Boiler and Pressure Vessel Code (Section III)** verification using the **Tresca criterion** for primary membrane stresses ($\sigma_{Tresca,m} \le S_m$) and primary + secondary stresses ($\sigma_{Tresca,tot} \le 3S_m$).
  * Fission gas release ($\text{Xe}$, $\text{Kr}$) and volatile impurities ($\text{N}_2$, $\text{H}_2\text{O}$) inventory at $60\text{ MWd/kgU}$ to size the minimum gas plenum volume ($V_{min}$) and height ($H_{plenum}$).

---

## ⚙️ Installation & Requirements

To run these scripts locally, make sure you have **Python 3.8+** installed. Clone this repository and install the required dependencies using `pip`:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
pip install -r requirements.txt
```

### Dependencies (`requirements.txt`)
* `numpy` — Array manipulation and vectorized numerical operations
* `scipy` — Numerical integration (`quad`), non-linear root finding (`fsolve`), and `.mat` file I/O
* `matplotlib` — 2D plotting of axial profiles
* `pyXSteam` — IAPWS-IF97 steam tables implementation for water and steam thermodynamic/transport properties

---

## 🚀 How to Run the Scripts

You can run the scripts directly from the terminal or inside **VS Code**:

1. **Assignment 1 (Natural Circulation):**
   ```bash
   python assignment1_1_c.py
   python assignment1_2.py
   ```
2. **Assignment 2 (Hot Channel & Fuel Rod Thermal Analysis):**
   ```bash
   python ass_2_finale.py
   ```
   > ⚠️ **Note:** Run `ass_2_finale.py` **before** `ass3_2.py`, as it generates the `.mat` data files (`Tci_data.mat`, `Tco_data.mat`, `p_in_data.mat`) required as input for the thermo-mechanical verification.

3. **Assignment 3 (Thermo-Mechanical Verification & Plenum Sizing):**
   ```bash
   python ass3_2.py
   ```

---

## 👤 Author

* **Master's Degree Student in Nuclear Engineering**
* Feel free to reach out or open an issue for questions, suggestions, or collaborations!"# Mix-codes" 
