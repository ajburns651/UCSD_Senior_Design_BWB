# ────────────────────────────────────────────────
# 155A BWB - Multi Design Iteration Script
# ────────────────────────────────────────────────
import openvsp as vsp
import numpy as np
import pandas as pd
import os

# --- MOVE IMPORTS TO TOP (Speedup: Prevents re-loading modules 50 times) ---
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

    # Locate the wing geometry id values
    wing_id = vsp.FindGeomsWithName("BodyandWing")[0]
    vstabalizer_id = vsp.FindGeomsWithName("VStabalizer")[0]

    for i in range(len(spans)):
        vsp.SetParmVal(wing_id, "Span", f"XSec_{i+1}", spans[i])
        vsp.SetParmVal(wing_id, "Root_Chord", f"XSec_{i+1}", root_chords[i])
        vsp.SetParmVal(wing_id, "Tip_Chord", f"XSec_{i+1}", tip_chords[i])
        vsp.SetParmVal(wing_id, "Sweep", f"XSec_{i+1}", sweeps[i])
        vsp.SetParmVal(wing_id, "InLESweep", f"XSec_{i+1}", sweeps[i])
        vsp.SetParmVal(wing_id, "Dihedral", f"XSec_{i+1}", dihedrals[i])
        vsp.Update()

    basefilename = f"MC_Run_{run_id}"
    vspfilename = f"{basefilename}.vsp3"
    vsp.WriteVSPFile(vspfilename)

    # 2. RUN CALCULATIONS
    Sref, Swet, WS_areas, VStabalizer_area, MAC, AR = GrabParams.sizing(vspfilename, wing_id, vstabalizer_id, len(spans))
    CD0 = GrabParams.parasite(vspfilename, altitude, mach) # Replaced Obj['Altitude'], Obj['Mach']
    WS_trs = np.array([tip / root for root, tip in zip(root_chords, tip_chords)]) 

    # Compute Inputs
    b = sum(spans)*2
    Wing_wetted_Area = Swet/Sref * (WS_areas[2] + WS_areas[3]) * (M2Ft **2) # Wing Area (Last 2 Sections) (Ft^2)
    Fuselage_wetted_Area = Swet/Sref * (WS_areas[0] + WS_areas[1]) * (M2Ft **2) # Fuselage Area (First 2 Sections) (Ft^2)
    Wing_sweep = sweeps[2] # Outer Wing Sweep (deg)
    Wing_taper = WS_trs[2] # Outer Wing Taper
    Vt_area = 2*VStabalizer_area * (M2Ft **2) # Area of Vertical Tail (Ft^2)
    Length_tail = .55 * root_chords[0] * M2Ft # Aproximated (ft)
    Length_fuselage = root_chords[0] * M2Ft   # Feet
    Diameter_fuselage = (spans[0] + 1.15)*2 * M2Ft # Feet (Includes 1.15m cargo space) 
    Payload_lb = payload_N / Ntolb # Payload weight (lb)

    # Find Weights
    Weight_Distributions = Weights.estimate_aircraft_weights(plot_pie=False, export_csv=False, Sw=Wing_wetted_Area, AR=AR,lambda_outer_deg=Wing_sweep,taper=Wing_taper,V_mach=mach, Svt=Vt_area, Sf = Fuselage_wetted_Area, Lt_ft=Length_tail,L_ft=Length_fuselage,D_ft=Diameter_fuselage,payload_lb=Payload_lb,fuel_fraction=fuel_frac)

    # Compute Inputs
    rho = compute_density.compute(altitude)
    MTOW = Weight_Distributions['weights_N']['total']
    plotting_flag = False
    printing_flag = False

    # Get Aero Vals
    LD_cruise, CL_cruise, Alpha_cruise, CDi_cruise, CDw_cruise, CD_total_cruise = Aero_Driver.bwb_cruise_analysis(plotting_flag,printing_flag,AR,b,Diameter_fuselage,sweeps,root_chords,Swet/Sref,WS_areas,mach,CD0,rho,MTOW)

    # Compute Inputs
    Offsets = np.zeros(len(spans))
    Offsets[0] = 0.0  # root section starts at 0 (nose or front reference)
    for i in range(1, len(spans)):
        # Cumulative X offset due to previous sweep and span
        prev_span = spans[i-1]
        prev_sweep = sweeps[i-1]
        Offsets[i] = Offsets[i-1] + prev_span * np.tan(np.radians(prev_sweep))

    # Compute Center of Gravity, Neutral Point, and Static Margin
    CG = CGNPSM.compute_cg(Weight_Distributions['percentages'],WS_trs,root_chords,spans,sweeps,WS_areas*2,Offsets,MAC,mach,LD_cruise,range_m,tsfc)
    AVL.generate_avl_file(spans, root_chords, tip_chords, sweeps, dihedrals,f'{basefilename}.avl',basefilename,mach,Sref,MAC,b,CG[0])
    NP = AVL.get_neutral_point_from_avl(base_name = basefilename, alpha_cruise = Alpha_cruise,avl_executable = r"C:\Users\ajbur\OneDrive\Desktop\School\MAE FILES\BURNS\MAE 155A\BWB_Python\AVLFunctions\avl352.exe",timeout_seconds = 15)
    SM = 100 * (NP - CG)/MAC

    # Compute Cost
    plotting_flag = False
    Cost_per_hr = Cost.compute_per_hour_cost(plotting_flag,tsfc,LD_cruise,MTOW) # Compute Per Hour Cost

    # 3. CLEANUP TEMPORARY FILES
    extensions_to_delete = [".txt",".avl",".vsp3","_CompGeom.csv", "_CompGeom.txt", "_ParasiteBuildUp.csv"]
    for ext in extensions_to_delete:
        file_path = f"{basefilename}{ext}"
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except PermissionError: pass 

    # 4. RETURN DESIRED DATA
    return {
        "Run_ID": run_id,
        # Save Outputs
        "MTOW": MTOW,
        "L_D_Cruise": LD_cruise,
        "Range": range_m,
        "Static_Margin": SM,
        "Span_Total": sum(spans)*2,
        "Cost_Per_Hour": Cost_per_hr, 
        "Dimension Inputs": [spans, root_chords, tip_chords, sweeps, dihedrals], # Includes all varying parameters
        "Mission Inputs": [range_m], # Includes all varying parameters
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
    Nominal_Spans = np.array([4.46351676,  7.8394856,  23.3415505, 2.33815741])
    Nominal_Sweeps = np.array([62.03301726, 60.99887757, 43.07284845, 40.44181654])
    Nominal_Roots = np.array([44.47919744, 32.0062888,   8.04649305,  0.84268331])
    Nominal_Tips = np.array([32.0062888, 8.04649305,  0.84268331,  0.26620108])
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
    N_Simulations = 1000 
    Variance = 0.3                              # +/- 30% variation
    Range_Variance = [.6, .1]
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
