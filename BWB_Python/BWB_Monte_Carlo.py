# ────────────────────────────────────────────────
# 155A BWB - Multi Design Iteration Script
# ────────────────────────────────────────────────
import numpy as np
import openvsp as vsp
import pandas as pd
import os
from pathlib import Path
import shutil
from scipy.stats.qmc import LatinHypercube, scale

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
                   run_id, save_vsp):
    
    # 0. COMPUTE WEIGHT FRACTIONS
    weight_dict = gross_weight.compute_fractions(Wp=payload_N, R=range_m, LD=23.42,
                                                 cT=tsfc)  # Fixed LD to Baseline BWB Value
    fuel_frac = weight_dict["Ws_Wg"]

    # 1. UPDATE VSP GEOMETRY
    vsp.ClearVSPModel()
    vsp.ReadVSPFile('SeniorDesign.vsp3')

    wing_id = vsp.FindGeomsWithName("BodyandWing")[0]
    vstabalizer_id = vsp.FindGeomsWithName("VStabalizer")[0]

    tip_chords = np.append(root_chords[1:], tip_chords[2])

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

    basefilename = f"MC_Run_{run_id}"
    vspfilename = f"{basefilename}.vsp3"
    vsp.WriteVSPFile(vspfilename)

    # 2. RUN CALCULATIONS
    Sref, Swet, WS_areas, VStabalizer_area, MAC, AR, b = GrabParams.sizing(
        vspfilename, wing_id, vstabalizer_id, len(spans))

    CD0 = GrabParams.parasite(vspfilename, altitude, mach)
    WS_trs = np.array([tip / root for root, tip in zip(root_chords, tip_chords)])

    # --- DERIVED GEOMETRY ---
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
    q = .5 * rho * (mach * 295.1) ** 2
    MTOW = Weight_Distributions['weights_N']['total']

    LD_cruise, CL_cruise, Alpha_cruise, CDi_cruise, CDw_cruise, CD_total_cruise = \
        Aero_Driver.bwb_cruise_analysis(
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

    # Copy airfoil .dat files into the AVL working directory
    avl_dir = Path(f'{basefilename}.avl').resolve().parent
    for src_dir in [Path('./AVLFunctions'), Path('.')]:
        for dat in src_dir.glob('*.dat'):
            dest = avl_dir / dat.name
            if not dest.exists():
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

    # --- STRUCTURAL STRESS (Section 3) ---
    mat_allowable = 900 * (10 ** 6)  # Pa  (7075 aluminium alloy)  FIX: was 430*10^6 (XOR, not power)
    spar_cap_area = 0.0052  # m^2 (assumed)
    FOS = 1.5

    section3_moment = Root_moments[2]
    section3_rc = root_chords[2]
    section3_y = (.07 * section3_rc) / 2  # (t/c) * chord / 2
    moment_of_inertia = 4 * spar_cap_area * (section3_y ** 2)  # parallel-axis theorem
    section3_stress = (section3_moment * section3_y) / moment_of_inertia
    stress_ratio = (section3_stress * FOS) / mat_allowable  # > 1.0 means failure

    SM = 100 * (NP - CG) / MAC

    Cost_per_hr = Cost.compute_per_hour_cost(False, tsfc, LD_cruise, MTOW)

    # --- CLEANUP ---
    if save_vsp:
        extensions_to_delete = [
            '.txt', '.csv', '.run', '_np.txt', '_moment.txt', '.avl',
            '_CompGeom.csv', '_CompGeom.txt', '_ParasiteBuildUp.csv', '.vsp3' 
        ]
    else:
        extensions_to_delete = [
            '.txt', '.csv', '.run', '_np.txt', '_moment.txt', '.avl',
            '_CompGeom.csv', '_CompGeom.txt', '_ParasiteBuildUp.csv', '.vsp3' 
        ]

    for ext in extensions_to_delete:
        file_path = f"{basefilename}{ext}"
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except PermissionError:
                pass

    # ── Return everything the plotting script needs ──────────────────────────
    # Scalar geometric parameters are flattened here so the CSV is fully
    # readable by BWB_plot_results.py without any post-processing.
    return {
        # Identifiers
        "Run_ID": run_id,

        # Mission Level
        "Range_(Nmi)": range_m / 1852,
        "Fuel_Fraction": fuel_frac,

        # Geometry
        "Wingspan": b,  # computed full wingspan
        "Spans": spans,
        "Sweeps": sweeps,
        "Root_Chords": root_chords,
        "Tip_Chords": tip_chords,
        "Swet": Swet,
        "Sref": Sref,
        "Taper Ratios": WS_trs,
        "Wing Section Areas": WS_areas,

        # Aero 
        "L_D_Cruise": LD_cruise,
        "CD0": CD0,
        "CDi_Cruise": CDi_cruise,
        "CDw_Cruise": CDw_cruise,
        "CD_Total_Cruise": CD_total_cruise,
        "CL_Cruise": CL_cruise,
        "Alpha_Cruise": Alpha_cruise,
        "AR": AR,
        "Neutral Point": NP,

        # Structural
        "MTOW": MTOW,
        "Static_Margin": SM,
        "Section3_Stress_Pa": section3_stress,
        "Stress_Ratio": stress_ratio,  # > 1.0 → structural failure
        "Root_Moment_Sec3": Root_moments[2],
        "Center of Gravity": CG,

        # Other
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
        data = analyze_design(
            spans=sim_spans, root_chords=sim_roots, tip_chords=sim_tips,
            sweeps=sim_sweeps, dihedrals=nom_dihedrals,
            range_m=sim_range, mach=sim_mach, payload_N=sim_payload,
            tsfc=sim_tsfc, altitude=nom_alt, run_id=i, save_vsp=False)
        print(f"Run {i} Complete")
        return data
    except Exception as e:
        print(f"Run {i} failed: {e}")
        return None


# ────────────────────────────────────────────────
# 2. MONTE CARLO LOOP (MAIN THREAD)
# ────────────────────────────────────────────────
if __name__ == '__main__':

    # Initial Conditions
    Nominal_Spans = np.array([4.18, 7.49, 20.87])
    Nominal_Roots = np.array([44.89, 32.98, 7.86])
    Nominal_Tips = np.array([32.98, 7.86, 1.70])
    Nominal_Sweeps = np.array([65, 57.46, 43.76])
    Nominal_Dihedrals = np.array([0.00, 0.00, 8.00])

    # ── NOMINAL MISSION / PERFORMANCE ───────────────────────────────────────
    Nominal_Range = 7000 * 1852  # Nmi → metres
    Nominal_Mach = 0.85
    Nominal_Altitude = 40000  # ft  (held constant)
    Nominal_Payload = 392000  # N
    Nominal_TSFC = 1.415e-5

    # ── GEOMETRY CONSTRAINTS ────────────────────────────────────────────────
    MIN_rootcs = np.array([43.0, 31.18, 0])  # passenger/cargo fit
    MIN_tipcs = np.array([31.18, 0, 0])
    MAX_sweeps = np.array([65, 65, 45])

    # ── SIMULATION CONFIG ───────────────────────────────────────────────────
    N_Simulations = 1000
    Variance = 0.3  # ± 30 % on geometry

    # ── STEP A: PRE-GENERATE ALL RANDOMISED INPUTS ──────────────────────────
    N_DIMS = 5
    sampler = LatinHypercube(d=N_DIMS, seed=42)  # seed for reproducibility
    samples = sampler.random(n=N_Simulations)  # shape: (N_Simulations, 7), all in [0, 1]

    l_bounds = np.array([1 - Variance,  # span
                         1 - Variance,  # sweep[0]
                         1 - Variance,  # sweep[1]
                         1 - Variance,  # sweep[2]
                         1])  # chord (can only grow)

    u_bounds = np.array([1 + Variance,  # span
                         1 + Variance,  # sweep[0]
                         1 + Variance,  # sweep[1]
                         1 + Variance,  # sweep[2]
                         1 + Variance])  # chord

    bad = np.where(l_bounds >= u_bounds)[0]
    if len(bad) > 0:
        for idx in bad:
            print(f"  Collapsed bound at index {idx}: lb={l_bounds[idx]:.4f} >= ub={u_bounds[idx]:.4f}")
        # Add a small epsilon floor to guarantee lb < ub
        u_bounds = np.where(l_bounds >= u_bounds, l_bounds + 1e-6, u_bounds)

    scaled = scale(samples, l_bounds, u_bounds)

    tasks = []
    for i in range(N_Simulations):
        span_factor = scaled[i, 0]
        sweep_factor = scaled[i, 1:4]
        chord_factor = scaled[i, 4]

        sim_spans = Nominal_Spans * span_factor
        sim_sweeps = np.minimum(Nominal_Sweeps * sweep_factor, MAX_sweeps)
        sim_roots = np.maximum(Nominal_Roots * chord_factor, MIN_rootcs)
        sim_tips = np.maximum(np.append(sim_roots[1:], Nominal_Tips[-1] * chord_factor), MIN_tipcs)
        sim_range = Nominal_Range

        task = (i, sim_spans, sim_roots, sim_tips, sim_sweeps, Nominal_Dihedrals,
                sim_range, Nominal_Mach, Nominal_Payload, Nominal_TSFC, Nominal_Altitude)
        tasks.append(task)

    # ── STEP B: SETUP PARALLEL POOL ─────────────────────────────────────────
    num_cores = max(1, multiprocessing.cpu_count() - 4)
    print(f"Starting Monte Carlo ({N_Simulations} runs) on {num_cores} CPU cores...")

    backup_interval = 100
    backup_filename = "Monte_Carlo_Backup_Temp.csv"
    results_log = []

    # ── STEP C: EXECUTE WITH PERIODIC BACKUPS ───────────────────────────────
    # maxtasksperchild=1 guards against VSP / AVL memory leaks between runs.
    with multiprocessing.Pool(processes=num_cores, maxtasksperchild=1) as pool:
        for count, result in enumerate(pool.imap_unordered(run_simulation, tasks), 1):

            if result is not None:
                results_log.append(result)

            # Periodic backup
            if count % backup_interval == 0:
                pd.DataFrame(results_log).to_csv(backup_filename, index=False)
                print(f"  → Backup saved: {count}/{N_Simulations} runs complete "
                      f"({len(results_log)} successful so far)")

    # ── STEP D: FINAL SAVE ──────────────────────────────────────────────────
    final_filename = "Monte_Carlo_Results_Expanded.csv"
    df_out = pd.DataFrame(results_log)
    df_out.to_csv(final_filename, index=False)

    if os.path.exists(backup_filename):
        os.remove(backup_filename)

    print(f"\nAnalysis complete.  "
          f"Saved {len(results_log)}/{N_Simulations} successful runs → '{final_filename}'")
    print(f"Columns: {list(df_out.columns)}")
 