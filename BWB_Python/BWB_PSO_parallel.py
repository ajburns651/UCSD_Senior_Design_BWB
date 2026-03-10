# ============================================================
# BWB Particle Swarm Optimization Driver (PARALLEL READY)
# ============================================================

import numpy as np
import pyswarms as ps
import openvsp as vsp
import pandas as pd
import os
import multiprocessing as mp
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
Nominal_Spans = np.array([4.08, 6.50, 19.00, 2])
Nominal_Sweeps = np.array([62.00, 67.00, 37.00, 37.00])

Nominal_Roots = np.array([43.00, 31.18, 5.00, 3.00])
Nominal_wing_tip = np.array([0.999])
Nominal_Tips = np.append(Nominal_Roots[1:], Nominal_wing_tip[0])

Nominal_Dihedrals = np.array([0.00, 0.00, 8.00, 9.25])

# Mission
Nominal_Range = 7000 * 1852
Nominal_Mach = 0.85
Nominal_Altitude = 40000
Nominal_Payload = 392000
Nominal_Fuel_Frac = 0.36
Nominal_TSFC = 1.415e-5

# Constraints
MIN_rootcs = np.array([43.0, 31.18, 4, 1])
MIN_tipcs = np.array([31.18, 6, 1, .1])
MAX_sweeps = np.array([67, 67, 45, 45])

Variance = 0.30
Variance2 = 2

# ============================================================
# 2. BUILD PSO BOUNDS
# ============================================================

span_lb = np.array([4.08, 6.50, Nominal_Spans[2]* (1 - Variance), 0.1])
span_ub = Nominal_Spans * np.array([1 + Variance,1 + Variance,1 + Variance,5/Nominal_Spans[-1]])

sweep_lb = Nominal_Sweeps * (1 - Variance)
sweep_ub = MAX_sweeps

root_lb = MIN_rootcs
root_ub = Nominal_Roots * np.array([1 + Variance,1 + Variance,1 + Variance2,1 + Variance])

tip_lb = MIN_tipcs
tip_ub = Nominal_Tips * (1 + Variance)

lower_bounds = np.concatenate([span_lb, root_lb, tip_lb, sweep_lb])
upper_bounds = np.concatenate([span_ub, root_ub, tip_ub, sweep_ub])
bounds = (lower_bounds, upper_bounds)

N_VARS = len(lower_bounds)

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
    section3_y = (.1447 * section3_rc)/2 # t/c * chord length divided by 2
    moment_of_inertia = 2 * spar_cap_area * (section3_y ** 2)                         # 2 * area (Parallel axis theorem)

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
    LD_ref = 25.0  # reference objective scale

    # == Static Margin ==
    SM_takeoff = result["Static_Margin"][0]
    if SM_takeoff < 0:           # Hard: -1 unit → 2× L/D_ref
        penalty += LD_ref * 2.0 * (-SM_takeoff / 15)
    elif SM_takeoff > 15:        # Soft: nudge only
        penalty += LD_ref * 0.2 * ((SM_takeoff - 15) / 15)

    # == Bending Moments ==
    moments = result["Root Moments"]
    root_cs = result["Root Chords"]
    mat_allowable = 510e6  # Pa https://en.wikipedia.org/wiki/7075_aluminium_alloy
    spar_cap_area = 0.02       # m^2 (assumed???)
    FOS = 1.5                 # Factor Of Safety

    # Compute Stress in Section 3
    section3_moment = moments[2]
    section3_rc = root_cs[2]
    section3_y = (.07 * section3_rc)/2 # t/c * chord length divided by 2
    moment_of_inertia = 2 * spar_cap_area * (section3_y ** 2)                         # 2 * area (Parallel axis theorem)

    section3_stress = (section3_moment * section3_y ) / moment_of_inertia   # Stress = My/I
    stress_ratio = (section3_stress * FOS) / mat_allowable

    if stress_ratio > 1.0:       # Hard: 10% over → 1× L/D_ref
        penalty += LD_ref * 10.0 * (stress_ratio - 1.0)
    elif stress_ratio > 0.9:     # Soft warning band
        penalty += LD_ref * 0.5 * ((stress_ratio - 0.9) / 0.1)

    # == Folding Wing Constraint ==
    spans = result["Spans"]
    cumspan3 = np.sum(spans[0:3])
    span4 = spans[3]

    if cumspan3 > 32.5:         # Hard: 1m error → ~1× L/D_ref
        penalty += LD_ref * 2.0 * (abs(cumspan3 - 32.5) / 32.5)

    if span4 > 5:                # Hard
        penalty += LD_ref * 2.0 * ((span4 - 5) / 5)
    elif span4 > 4.5:              # Soft
        penalty += LD_ref * 0.1 * ((span4 - 3) / 3)
    
    return penalty


def evaluate_particle(args):
    particle, idx = args

    try:
        spans = particle[0:4]
        roots = particle[4:8]
        tips = particle[8:12]
        sweeps = particle[12:16]

        run_id = f"{os.getpid()}_{idx}_{uuid.uuid4().hex[:6]}"

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

        cost = - result["L_D_Cruise"]
        cost += constraint_penalty(result)

        return cost

    except Exception as e:
        print("Worker failed:", e)
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

def pso_cost_parallel(X):
    with mp.Pool(processes=N_PROCESSES) as pool:
        args = [(X[i], i) for i in range(X.shape[0])]
        costs = pool.map(evaluate_particle, args)

    return np.array(costs)


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
        n_particles=75,
        dimensions=N_VARS,
        options=options,
        bounds=bounds,
    )

    best_cost, best_pos = optimizer.optimize(
        pso_cost,
        iters=40,
        verbose=True,
    )

    # ========================================================
    # 6. REPORT RESULTS
    # ========================================================

    print("\n==============================")
    print("OPTIMIZATION COMPLETE")
    print("==============================")

    best_spans = best_pos[0:4]
    best_roots = best_pos[4:8]
    best_tips = best_pos[8:12]
    best_sweeps = best_pos[12:16]
    best_tips = np.append(best_roots[1:], best_tips[3])

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
