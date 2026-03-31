# ============================================================
# BWB Particle Swarm Optimization Driver (PARALLEL READY)
# ============================================================

import numpy as np
import pyswarms as ps
import openvsp as vsp
import pandas as pd
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid
from pathlib import Path
import shutil

from OpenVSPHooks import GrabParams
from WeightFunctions import CGNPSM, Weights
from AeroFunctions import Aero_Driver, compute_density
from AVLFunctions import AVL
from CostFunctions import Cost
from Marimo import gross_weight

USE_PARALLEL = True
N_PROCESSES = max(1, mp.cpu_count() // 2)


# DELETE OLD VSP
files = ["PSO_Run_Optimized_Plane_Design.avl", "PSO_Run_Optimized_Plane_Design.vsp3"]

for file in files:
    if os.path.exists(file):
        os.remove(file)

M2Ft = 3.28084
Ntolb = 4.44822

# ============================================================
# 1. NOMINAL VALUES
# ============================================================

# Geometry

Nominal_Spans = np.array([ 4.08373908,  7.44966705, 20.51247145])
Nominal_Sweeps = np.array([66.590491,   57.40578817, 44.59548356])
Nominal_Roots = np.array([46.18196837, 32.49980379,  7.57755233])
Nominal_Tips = np.array([32.49980379,  7.57755233,  1.67873103])
Nominal_Dihedrals = np.array([0.00, 0.00, 8.00])

# Mission
Nominal_Range = 7000 * 1852
Nominal_Mach = 0.85
Nominal_Altitude = 40000
Nominal_Payload = 392000
Nominal_Fuel_Frac = 0.36
Nominal_TSFC = 1.415e-5

# Constraints
MIN_rootcs = np.array([43.0, 31.18, 6.8])
MIN_tipcs = np.array([31.18, 6.8, 1.6])
MAX_sweeps = np.array([67, 67, 45])

Variance = 0.2
Variance2 = 0.2

# ============================================================
# 2. BUILD PSO BOUNDS
# ============================================================

span_lb = np.array([4.08, 6.50, Nominal_Spans[2]* (1 - Variance)])
span_ub = Nominal_Spans * np.array([1 + Variance,7.5/Nominal_Spans[1],1 + Variance])

sweep_lb = Nominal_Sweeps * (1 - Variance)
sweep_ub = MAX_sweeps

root_lb = MIN_rootcs
root_ub = Nominal_Roots * np.array([1 + Variance,1 + Variance,1 + Variance2])

tip_lb = MIN_tipcs
tip_ub = Nominal_Tips * (1 + Variance)

lower_bounds = np.concatenate([span_lb, root_lb, [MIN_tipcs[2]], sweep_lb])
upper_bounds = np.concatenate([span_ub, root_ub, [max(Nominal_Tips[2] * (1 + Variance), MIN_tipcs[2])], sweep_ub])
bounds = (lower_bounds, upper_bounds)

N_VARS = len(lower_bounds)

def analyze_design(spans, root_chords, tip_chords, sweeps, dihedrals,
                   range_m, mach, payload_N, tsfc, altitude,
                   run_id, save_vsp=False):

     # 0. COMPUTE WEIGHT FRACTIONS
    weight_dict = gross_weight.compute_fractions(Wp=payload_N, R=range_m, LD = 24, cT=tsfc) # Here we fix LD to our BCR value, and payload weight is constant
    fuel_frac = weight_dict["Ws_Wg"]

    # 1. UPDATE VSP GEOMETRY
    vsp.ClearVSPModel()
    vsp.ReadVSPFile('SeniorDesign.vsp3')

    wing_id = vsp.FindGeomsWithName("BodyandWing")[0]
    vstabalizer_id = vsp.FindGeomsWithName("VStabalizer")[0]

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
    Wing_wetted_Area = Swet / Sref * (WS_areas[2]) * (M2Ft ** 2)
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

    spar_cap_area = .0052       # m^2 (assumed???) 2*.02 = .04 4

    # Compute Stress in Section 3
    section3_moment = Root_moments[2]
    section3_rc = root_chords[2]
    section3_y = (.07 * section3_rc)/2 # t/c * chord length divided by 2
    moment_of_inertia = 4 * spar_cap_area * (section3_y ** 2)                         # 2 * area (Parallel axis theorem)

    section3_stress = (section3_moment * section3_y ) / moment_of_inertia   # Stress = My/I
 
    SM = 100 * (NP - CG) / MAC

    Cost_per_hr = Cost.compute_per_hour_cost(False, tsfc, LD_cruise, MTOW)

    # --- CLEANUP ---
    if save_vsp:
        extensions_to_delete = [
            ".txt", ".csv", ".run", '_CompGeom.csv', '_CompGeom.txt', '_moment.txt', '_np.txt', '_ParasiteBuildUp.csv'
        ]
    else:
        extensions_to_delete = [
            ".txt", ".avl", ".csv", ".run", '_CompGeom.csv', '_CompGeom.txt', '_moment.txt', '_np.txt','_ParasiteBuildUp.csv', ".vsp3"
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
        "Spans": spans,
        "Cost_Per_Hour": Cost_per_hr,
    }

def constraint_penalty(result):
    penalty = 0.0
    LD_ref = 15.0  # Scaling Objective

    # == Static Margin ==
    SM_takeoff = result["Static_Margin"][0]
    if SM_takeoff < 8:           # Hard: -1 unit → 2× L/D_ref
        penalty += LD_ref * 0.2 * (8 - SM_takeoff) / 8
    elif SM_takeoff > 15:        # Soft: nudge only
        penalty += LD_ref * 0.2 * ((SM_takeoff - 15) / 15)

    # == Bending Moments ==
    mat_allowable = 900*(10**6)  # Pa https://en.wikipedia.org/wiki/7075_aluminium_alloy
    FOS = 1.5                 # Factor Of Safety

    # Compute Stress Ratio in Section 3
    stress_ratio = (result["Section 3 Stress"] * FOS) / mat_allowable

    if stress_ratio > 1.0:       # Hard: 10% over → 1× L/D_ref
        penalty += LD_ref * 2.0 * (stress_ratio - 1.0)/.1

    # == Folding Wing Constraint ==
    spans = result["Spans"]
    cumspan3 = np.sum(spans[0:3])

    #if cumspan3 > 32.5:         # Hard: 1m error → ~1× L/D_ref
    #    penalty += LD_ref * 2.0 * (abs(cumspan3 - 32.5) / 32.5)
    
    return penalty

import traceback
def evaluate_particle(args):
    particle, idx = args

    try:
        spans = particle[0:3]
        roots = particle[3:6]
        outer_tip = particle[6]
        sweeps = particle[7:10]

        run_id = f"{os.getpid()}_{idx}_{uuid.uuid4().hex[:6]}"

        tips = np.array([roots[1], roots[2], outer_tip])

        result = analyze_design(
        spans=spans,
        root_chords=roots,
        tip_chords=tips,
        sweeps=sweeps,
        dihedrals=Nominal_Dihedrals,
        range_m=Nominal_Range,
        mach=Nominal_Mach,
        payload_N=Nominal_Payload,
        tsfc=Nominal_TSFC,
        altitude=Nominal_Altitude,
        run_id=run_id,
        save_vsp=False)

        L_D = result["L_D_Cruise"]
        #cost_scale = (25 - L_D) * abs(25 - L_D)

        cost = -L_D + constraint_penalty(result)
        print(f"L/D: {result['L_D_Cruise']:.2f}  |  penalty: {constraint_penalty(result):.2f}  |  cost: {cost:.2f}")

        return cost

    except Exception as e:

        print("Worker failed:", e)
        traceback.print_exc()
        return 1e15


# ============================================================
# SERIAL COST (debug mode)
# ============================================================

def pso_cost_serial(X):
    costs = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        costs[i] = evaluate_particle((X[i], i))
    return costs


# ============================================================
# PARALLEL COST
# ============================================================

# Create once at module level — workers stay alive entire run
_executor = None

def get_executor():
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=N_PROCESSES)
    return _executor

def pso_cost_parallel(X):
    executor = get_executor()
    futures = {executor.submit(evaluate_particle, (X[i], i)): i 
               for i in range(X.shape[0])}
    
    costs = np.zeros(X.shape[0])
    for future in as_completed(futures):
        i = futures[future]
        try:
            costs[i] = future.result()
        except Exception as e:
            print(f"Particle {i} failed: {e}")
            costs[i] = 1e6
    
    return costs


# ============================================================
# MASTER SWITCH
# ============================================================

def pso_cost(X):
    if USE_PARALLEL:
        return pso_cost_parallel(X)
    else:
        return pso_cost_serial(X)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    mp.freeze_support()

    print("Parallel:", USE_PARALLEL)
    print("Processes:", N_PROCESSES)

    options = {"c1": 1, "c2": 2.0, "w": 0.5}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=25,
        dimensions=N_VARS,
        options=options,
        bounds=bounds,
    )

    with ProcessPoolExecutor(max_workers=N_PROCESSES) as executor:

        def pso_cost_parallel(X):
            futures = {executor.submit(evaluate_particle, (X[i], i)): i
                       for i in range(X.shape[0])}
            costs = np.zeros(X.shape[0])
            for future in as_completed(futures):
                i = futures[future]
                try:
                    costs[i] = future.result()
                except Exception as e:
                    print(f"Particle {i} failed: {e}")
                    costs[i] = 1e6
            return costs

        def pso_cost(X):
            if USE_PARALLEL:
                return pso_cost_parallel(X)
            else:
                return pso_cost_serial(X)

        try:
            best_cost, best_pos = optimizer.optimize(
                pso_cost,
                iters=15,
                verbose=True,
            )
        except Exception as e:
            print(f"Optimization failed: {e}")
            raise

    # ========================================================
    # REPORT RESULTS
    # ========================================================

    print("\n==============================")
    print("OPTIMIZATION COMPLETE")
    print("==============================")

    best_spans     = best_pos[0:3]
    best_roots     = best_pos[3:6]
    best_outer_tip = best_pos[6]
    best_sweeps    = best_pos[7:10]
    best_tips      = np.array([best_roots[1], best_roots[2], best_outer_tip])

    best_result = analyze_design(
        spans=best_spans,
        root_chords=best_roots,
        tip_chords=best_tips,
        sweeps=best_sweeps,
        dihedrals=Nominal_Dihedrals,
        range_m=Nominal_Range,
        mach=Nominal_Mach,
        payload_N=Nominal_Payload,
        tsfc=Nominal_TSFC,
        altitude=Nominal_Altitude,
        run_id='Optimized_Plane_Design',
        save_vsp=True)

    print("\nBest Design:")
    print("Spans:", best_spans)
    print("Roots:", best_roots)
    print("Tips:", best_tips)
    print("Sweeps:", best_sweeps)
    for key, value in best_result.items():
        print(f"{key}: {value}")