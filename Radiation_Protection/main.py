import yaml
import numpy as np
import pandas as pd

## import dati
Nrods = 126
with open("dati.yaml", "r", encoding="utf-8") as f:
    dati = yaml.safe_load(f)

Dfuel = np.linspace(dati["Fuel"]["Din_min"], dati["Fuel"]["Din_max"], 3)

rho = np.linspace(dati["Fuel"]["rho_min"], dati["Fuel"]["rho_max"], 5)

A_pellet = (np.pi / 4.0) * (dati['Fuel']['Dout']**2 - Dfuel**2)
Vf = A_pellet * dati['Fuel']['h_stack'] * Nrods
    
#massa molare 
e=0.04
MM_U = 235*e + (1-e)*238
fU = MM_U / (dati['Fuel']['MMUO2'])

res = {}
for r in rho: 
    mass = []
    for V in Vf:
        m = V * r * fU / 1000 # kg
        mass.append(m)
    res[f"rho = {r:.2f} g/cm3"] = mass
    
df_mass = pd.DataFrame(res, index=[f"d_in = {d*10:.1f} mm" for d in Dfuel])
print(100*"=")
print("Massa totale di Uranio nell'Assembly [kg] (Rif. IAEA Table IV: 119.7 kg):")
print(df_mass)
print(f"\n")
print(100*"=")
print(f"Volume totale di fuel con r_in minimo e pho minima: {Vf[0]:.2f} cm^3 ")
print(f"\n")


#file_activity = 'attivia_kola3_microshield.csv'

#Ass: 1) assembly omogeneo (e=cosnt)
mass_U = df_mass.iloc[0, 0] # kg
Vu = Vf[0] # cm^3
Au = A_pellet[0] * Nrods # cm^2
rho_u = rho[0]
print(f"Massa totale di fuel con r_in minimo e pho minima: {mass_U:.2f} kg ")
print(f"\n")

sfcompo_kola3_s5 = {
    "U-234": 2.000e-4,
    "U-235": 1.569e-2,
    "U-236": 5.500e-3,
    "U-238": 9.345e-1,
    "Pu-238": 1.970e-4,
    "Pu-239": 5.819e-3,
    "Pu-240": 1.977e-3,
    "Pu-241": 7.920e-4,
    "Pu-242": 3.360e-4,
    "Nd-143": 9.305e-4,  # Stabile
    "Nd-145": 7.365e-4,  # Stabile
    "Nd-146": 7.260e-4,  # Stabile
    "Nd-148": 3.860e-4,  # Stabile
    "Cm-242": 5.800e-6,
    "Cm-244": 1.100e-5,
    "Am-241": 5.300e-4,
    "Am-243": 1.000e-4,
    "Cs-133": 1.341e-3,  # Stabile
    "Cs-134": 6.800e-6,
    "Cs-135": 7.724e-4,
    "Cs-137": 1.097e-3,
    "Ce-144": 6.300e-8,
    "Eu-151": 5.000e-6,  # Praticamente stabile (T_1/2 ~ 5e18 y)
    "Eu-153": 1.250e-4,  # Stabile
    "Eu-154": 1.300e-5,
    "Eu-155": 2.330e-6,
    "Sm-147": 2.280e-4,
    "Sm-149": 2.900e-6,  # Stabile
    "Sm-150": 2.480e-4,  # Stabile
    "Sm-151": 1.260e-5,
    "Sm-152": 9.900e-5,  # Stabile
    "Gd-155": 3.000e-6,  # Stabile
    "Mo-95": 7.000e-3,   # Stabile
    "Pd-105": 4.900e-4,  # Stabile
    "Pd-108": 7.000e-5,  # Stabile
    "Ag-109": 6.000e-5,  # Stabile
    "Ru-101": 6.000e-5,  # Stabile
    "Np-237": 6.650e-4
}

nrc_specific_activity = {
    # Isotopo: (TBq/g, Ci/g)
    "U-234":  (2.3e-4, 6.2e-3),
    "U-235":  (8.0e-8, 2.2e-6),
    "U-236":  (2.4e-6, 6.5e-5),
    "U-238":  (1.2e-8, 3.4e-7),
    "Pu-238": (6.3e-1, 1.7e+1),
    "Pu-239": (2.3e-3, 6.2e-2),
    "Pu-240": (8.4e-3, 2.3e-1),
    "Pu-241": (3.8e+0, 1.0e+2),
    "Pu-242": (1.5e-4, 3.9e-3),
    "Cm-242": (1.2e+2, 3.3e+3),
    "Cm-244": (3.0e+0, 8.1e+1),
    "Am-241": (1.3e-1, 3.4e+0),
    "Am-243": (7.4e-3, 2.0e-1),
    "Cs-134": (4.8e+1, 1.3e+3),
    "Cs-135": (4.3e-5, 1.2e-3),
    "Cs-137": (3.2e+0, 8.7e+1),
    "Ce-144": (1.2e+2, 3.2e+3),
    "Eu-154": (9.8e+0, 2.6e+2),
    "Eu-155": (1.8e+1, 4.9e+2),
    "Sm-147": (8.5e-10, 2.3e-8),
    "Sm-151": (9.7e-1, 2.6e+1),
    "Np-237": (2.6e-5, 7.1e-4)
}

data=[]; act_tot=0
act_per_iso = {}

for iso, conc in sfcompo_kola3_s5.items():
    if iso in nrc_specific_activity:
        sa_TBq_g, sa_Ci_g = nrc_specific_activity[iso]
        sa_Bq_g = sa_TBq_g * 1e12  # Bq
        
        mass_iso_g = conc * mass_U *1e3 #g
        act_Bq = mass_iso_g * sa_Bq_g
        act_Ci = mass_iso_g * sa_Ci_g
        print(f"Isotopo: {iso} Attività (Bq): {act_Bq:.4e}")
        
        act_per_iso[iso] = act_Bq
        
        data.append({
            "Isotopo": iso,
            "Conc [g/gUi]": conc,
            "Massa [g]": mass_iso_g,
            "Spec_Act [Bq/g]": sa_Bq_g,
            "Spec_Act [Ci/g]": sa_Ci_g,
            "Attività [Bq]": act_Bq,
            "Attività [Ci]": act_Ci
        })
        
        act_tot = act_tot + act_Bq


#act_per_iso["Ba-137m"] = 0.946 * act_per_iso["Cs-137"]  
#act_per_iso["Pr-144"]  = act_per_iso["Ce-144"]           

with open("origen_ms.txt", "w", encoding="ascii", newline="\r\n") as f:
    f.write("Activity_KOLA_3_Ci\n")
    for iso, a_bq in act_per_iso.items():
        a_ci = a_bq / 3.7e10
        nuc = iso.upper().replace("-", "")
        f.write(f"{nuc} {a_ci:.10f}\n".replace(".", ","))
        
print(f"Attività totale dell'assembly Kola-3: {act_tot:.3e} Bq ({act_tot/3.7e10:.3e} Ci)")
print(f"\n")

Aexa = np.sqrt(3)/2*dati['Assembly']['W']**2
R_assembly = np.sqrt(Aexa/np.pi)

print(f"Raggio equivalente dell'assembly: {R_assembly:.2f} cm")
print(f"\n")

## densità effettive
rhof_eff = Au / Aexa * rho_u; rhoh2o_eff = (Aexa - Au) / Aexa * dati['Coolant']['rho']

print(f"Densità effettive: fuel = {rhof_eff:.2f} g/cm3, H2O = {rhoh2o_eff:.2f} g/cm3")
print(f"\n")