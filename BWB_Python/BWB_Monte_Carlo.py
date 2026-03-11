# ────────────────────────────────────────────────
# 155A BWB - Multi Design Iteration Script
# ────────────────────────────────────────────────
import numpy as np
import openvsp as vsp
import pandas as pd
import os
from pathlib import Path
import shutil

from OpenVSPHooks import GrabParams
from WeightFunctions import CGNPSM, Weights
from AeroFunctions import Aero_Driver, compute_density
from AVLFunctions import AVL
from CostFunctions import Cost
from Marimo import gross_weight

# ────────────────────────────────────────────────
# Section 0 - Define Frequently Used Constants
# ────────────────────────────────────────────────
M2Ft = 3.28084
Ntolb = 4.44822

def analyze_design(spans, root_chords, tip_chords, sweeps, dihedrals,
                   range_m, mach, payload_N, tsfc, altitude,
                   run_id, save_vsp=False):

     # 0. COMPUTE WEIGHT FRACTIONS
    weight_dict = gross_weight.compute_fractions(Wp=payload_N, R=range_m, LD = 23.42, cT=tsfc) # Here we fix LD to our BCR value, and payload weight is constant
    fuel_frac = weight_dict["Ws_Wg"]

    # 1. UPDATE VSP GEOMETRY
    vsp.ClearVSPModel()
    vsp.ReadVSPFile('SeniorDesign.vsp3')

    wing_id = vsp.FindGeomsWithName("BodyandWing")[0]
    vstabalizer_id = vsp.FindGeomsWithName("VStabalizer")[0]

    tip_chords = np.append(root_chords[1:], tip_chords[3])

    # --- UPDATE GEOMETRY ---
    for i in range(len(spans)):
        vsp.SetParmVal(wing_id, "Span", f"XSec_{i + 1}", spans[i])
        vsp.SetParmVal(wing_id, "Root_Chord", f"XSec_{i + 1}", root_chords[i])
        vsp.SetParmVal(wing_id, "Tip_Chord", f"XSec_{i + 1}", tip_chords[i])
        vsp.SetParmVal(wing_id, "Sweep", f"XSec_{i + 1}", sweeps[i])
        vsp.SetParmVal(wing_id, "Dihedral", f"XSec_{i + 1}", dihedrals[i])
        vsp.Update()

    # Create Matching Angles For Outward Wings
    vsp.SetParmVal(wing_id, "InLESweep", f"XSec_3", sweeps[2])
    vsp.Update()

    basefilename = f"PSO_Run_{run_id}"
    vspfilename = f"{basefilename}.vsp3"
    vsp.WriteVSPFile(vspfilename)

    # 2. RUN CALCULATIONS
    Sref, Swet, WS_areas, VStabalizer_area, MAC, AR, b = GrabParams.sizing(
        vspfilename, wing_id, vstabalizer_id, len(spans))

    CD0 = GrabParams.parasite(vspfilename, altitude, mach)
    WS_trs = np.array([tip / root for root, tip in zip(root_chords, tip_chords)])

    # --- DERIVED GEOMETRY ---
    # Compute Wingspan (Due to sweep)
    Wing_wetted_Area = Swet / Sref * (WS_areas[2] + WS_areas[3]) * (M2Ft ** 2)
    Fuselage_wetted_Area = Swet / Sref * (WS_areas[0] + WS_areas[1]) * (M2Ft ** 2)
    Wing_sweep = sweeps[2]
    Wing_taper = WS_trs[2]
    Vt_area = 2 * VStabalizer_area * (M2Ft ** 2)
    Length_tail = .55 * root_chords[0] * M2Ft
    Length_fuselage = root_chords[0] * M2Ft
    Diameter_fuselage = (spans[0] + 1.15) * 2 * M2Ft
    Payload_lb = payload_N / Ntolb

    # --- WEIGHTS ---
    Weight_Distributions = Weights.estimate_aircraft_weights(
        plot_pie=False, export_csv=False,
        Sw=Wing_wetted_Area, AR=AR,
        lambda_outer_deg=Wing_sweep,
        taper=Wing_taper,
        V_mach=mach,
        Svt=Vt_area,
        Sf=Fuselage_wetted_Area,
        Lt_ft=Length_tail,
        L_ft=Length_fuselage,
        D_ft=Diameter_fuselage,
        payload_lb=Payload_lb,
        fuel_fraction=fuel_frac)

    rho = compute_density.compute(altitude)
    q = .5 * rho * (mach * 295.1)**2
    MTOW = Weight_Distributions['weights_N']['total']

    LD_cruise, CL_cruise, Alpha_cruise, CDi_cruise, CDw_cruise, CD_total_cruise = Aero_Driver.bwb_cruise_analysis(
                                                                    False, False, AR, b, Diameter_fuselage, sweeps,
                                                                    root_chords, Swet / Sref, WS_areas, mach, CD0, rho, MTOW)

    # --- OFFSETS ---
    Offsets = np.zeros(len(spans))
    for i in range(1, len(spans)):
        Offsets[i] = Offsets[i - 1] + spans[i - 1] * np.tan(np.radians(sweeps[i - 1]))

    CG = CGNPSM.compute_cg(
        Weight_Distributions['percentages'], WS_trs,
        root_chords, spans, sweeps,
        WS_areas * 2, Offsets, MAC, mach,
        LD_cruise, range_m, tsfc)

    # After generate_avl_file, copy airfoil files into the same dir as the .avl
    avl_dir = Path(f'{basefilename}.avl').resolve().parent
    for src_dir in [Path('./AVLFunctions'), Path('.')]:
        for dat in src_dir.glob('*.dat'):
            dest = avl_dir / dat.name
            if not dest.exists():  # skip if already there
                try:
                    shutil.copy2(dat, dest)
                except (OSError, shutil.Error):
                    pass  # another worker already copied it, that's fine

    AVL.generate_avl_file(
        spans, root_chords, tip_chords, sweeps, dihedrals,
        f'{basefilename}.avl', basefilename,
        mach, Sref, MAC, b, CG[0])

    NP = AVL.get_neutral_point_from_avl(
        base_name=basefilename,
        alpha_cruise=Alpha_cruise,
        CL_cruise=CL_cruise,
        rho=rho,
        avl_executable=r".\AVLFunctions\avl352.exe",
        timeout_seconds=15)
    
    normalized_moments = AVL.get_root_moment_from_avl(
        base_name=basefilename,
        alpha_cruise=Alpha_cruise,
        CL_cruise=CL_cruise,
        WS_spans=spans,
        rho=rho,
        avl_executable=r".\AVLFunctions\avl352.exe",
        timeout_seconds=15)
    Root_moments = normalized_moments * Sref * b * q

    # Compute Section 3 Stress
    mat_allowable = 510*(10**6)  # Pa https://en.wikipedia.org/wiki/7075_aluminium_alloy
    spar_cap_area = .02       # m^2 (assumed???)
    FOS = 1.5                 # Factor Of Safety

    # Compute Stress in Section 3
    section3_moment = Root_moments[2]
    section3_rc = root_chords[2]
    section3_y = (.07 * section3_rc)/2 # t/c * chord length divided by 2
    moment_of_inertia = 2 * spar_cap_area * (section3_y ** 2)                         # 2 * area (Parallel axis theorem)

    section3_stress = (section3_moment * section3_y ) / moment_of_inertia   # Stress = My/I
 
    SM = 100 * (NP - CG) / MAC

    Cost_per_hr = Cost.compute_per_hour_cost(False, tsfc, LD_cruise, MTOW)

    # --- CLEANUP ---
    if save_vsp:
        extensions_to_delete = [
            ".txt", ".csv", ".run", '_CompGeom.csv', '_CompGeom.txt', '_np.txt', '_ParasiteBuildUp.csv'
        ]
    else:
        extensions_to_delete = [
            ".txt", ".avl", ".csv", ".run", '_CompGeom.csv', '_CompGeom.txt', '_np.txt','_ParasiteBuildUp.csv', ".vsp3"
        ]

    for ext in extensions_to_delete:
        file_path = f"{basefilename}{ext}"
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                pass

    return {
        "Run_ID": run_id,
        "MTOW": MTOW,
        "L_D_Cruise": LD_cruise,
        "Range (Nmi)": range_m / 1852,
        "Static_Margin": SM,
        "Wingspan": b,
        "Root Moments": Root_moments,
        "Section 3 Stress": section3_stress,
        "Root Chords": root_chords,
        "Cost_Per_Hour": Cost_per_hr,
    }


# ────────────────────────────────────────────────
# 1. PARALLEL WRAPPER FUNCTION
# ────────────────────────────────────────────────
import multiprocessing
def run_simulation(args):
    """Unpacks the task arguments and runs the analysis."""
    (i, sim_spans, sim_roots, sim_tips, sim_sweeps, nom_dihedrals, 
     sim_range, sim_mach, sim_payload, sim_tsfc, nom_alt) = args
    
    try:
        print(f"Run {i} Starting...")
        data = analyze_design(
            spans=sim_spans, root_chords=sim_roots, tip_chords=sim_tips, 
            sweeps=sim_sweeps, dihedrals=nom_dihedrals, 
            range_m=sim_range, mach=sim_mach, payload_N=sim_payload, 
            tsfc=sim_tsfc, altitude=nom_alt, run_id=i)
        print(f"Run {i} Complete")
        return data
    except Exception as e:
        print(f"Run {i} failed: {e}")
        return None

# ────────────────────────────────────────────────
# 2. MONTE CARLO LOOP (MAIN THREAD)
# ────────────────────────────────────────────────
if __name__ == '__main__':
    # --- DEFINE NOMINALS ---
    # Geometry
    Nominal_Spans = np.array([4.08654,  7.28272,  17.10197, 4.05377])
    Nominal_Sweeps = np.array([62.47886, 64.38714, 40.23984, 40.03606])

    Nominal_Roots = np.array([43.57478, 32.33918,   6.68997,  1.16213])
    Nominal_Tips = np.array([32.33918, 6.68997,  1.16213,  0.49380])
    Nominal_Dihedrals = np.array([0.00, 0.00, 8.00, 9.25])

    # Mission/Performance
    Nominal_Range = 7000 * 1852       # Nmi -> Meters (Varied)
    Nominal_Mach = 0.85
    Nominal_Altitude = 40000          # Kept constant in this example, but can vary too
    Nominal_Payload = 392000          # Newtons
    Nominal_TSFC = 1.415 * 10**(-5)

    # --- DEFINE CONSTRAINTS ---
    MIN_rootcs = np.array([43.0, 31.18, 0, 0])   # Constraint so that Passengers/Cargo Can fit
    MIN_tipcs  = np.array([31.18, 0, 0, 0])      # Constraint so that Passengers/Cargo Can fit
    MAX_sweeps = np.array([65, 65, 45, 45])      # Constraint to get feasible geometry

    # --- CONFIGURE SIMULATION ---
    N_Simulations = 1 
    Variance = 0.0                              # +/- 30% variation
    Range_Variance = [0, 0]

    # Step A: Pre-generate all the randomized inputs (tasks)
    tasks = []
    for i in range(N_Simulations):
        span_factor = np.random.uniform(1-Variance, 1+Variance)
        sweep_factor = np.random.uniform(1-Variance, 1+Variance, size=len(Nominal_Sweeps))
        chord_factor = np.random.uniform(1, 1+Variance) # Chord can not decrease from current

        sim_spans = Nominal_Spans * span_factor
        sim_sweeps = np.minimum(Nominal_Sweeps * sweep_factor, MAX_sweeps)
        sim_roots = np.maximum(Nominal_Roots * chord_factor, MIN_rootcs)
        sim_tips = np.maximum(np.append(sim_roots[1:], Nominal_Tips[-1] * chord_factor), MIN_tipcs)
        
        range_factor = np.random.uniform(1-Range_Variance[0], 1+Range_Variance[1])

        sim_range = Nominal_Range * range_factor
        sim_mach = Nominal_Mach
        sim_payload = Nominal_Payload
        sim_tsfc = Nominal_TSFC
        
        # Package everything into a single tuple for this specific run
        task = (i, sim_spans, sim_roots, sim_tips, sim_sweeps, Nominal_Dihedrals, 
                sim_range, sim_mach, sim_payload, sim_tsfc, Nominal_Altitude)
        tasks.append(task)

    # Step B: Setup the Parallel Pool
    # We use (cpu_count - 1) or (cpu_count - 2) so your computer doesn't totally freeze up 
    # while the Monte Carlo runs in the background.
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"Starting Monte Carlo on {num_cores} CPU cores...")

    # --- NEW: Setup our backup parameters ---
    backup_interval = 100 # Save a backup every 100 completed runs
    backup_filename = "Monte_Carlo_Backup_Temp.csv"
    results_log = []

    # Step C: Execute in Parallel with imap_unordered
    # (Keeping maxtasksperchild=1 to protect you from RAM memory leaks!)
    with multiprocessing.Pool(processes=num_cores, maxtasksperchild=1) as pool:
        
        # pool.imap_unordered hands back results as soon as they finish.
        # enumerate(..., 1) just gives us a handy counter starting at 1.
        for count, result in enumerate(pool.imap_unordered(run_simulation, tasks), 1):
            
            # Filter out the failed runs
            if result is not None:
                results_log.append(result)
                
            # Step D: The Periodic Backup
            if count % backup_interval == 0:
                pd.DataFrame(results_log).to_csv(backup_filename, index=False)
                print(f"-> Backup updated: {count}/{N_Simulations} runs complete...")

    # Step E: Final Save
    final_filename = "Monte_Carlo_Results_Expanded.csv"
    pd.DataFrame(results_log).to_csv(final_filename, index=False)

    # Optional: Delete the temporary backup file now that we have the final one
    if os.path.exists(backup_filename):
        os.remove(backup_filename)

    print(f"Analysis complete. Successfully saved {len(results_log)}/{N_Simulations} runs.")
