# ────────────────────────────────────────────────
# 155A BWB - Particle Swarm Optimization for max L/D cruise
# CHANGES FROM ORIGINAL:
#   1. Saves best VSP file whenever a new global best is found
#   2. Saves full geometry (spans, chords, sweeps, dihedrals) to CSV each iteration
#   3. Prints full reconstructible geometry at final summary
# ────────────────────────────────────────────────
import openvsp as vsp
import numpy as np
import pandas as pd
import os
import json

from OpenVSPHooks import GrabParams
from WeightFunctions import CGNPSM, Weights
from AeroFunctions import Aero_Driver, compute_density
from AVLFunctions import AVL
from CostFunctions import Cost

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# ────────────────────────────────────────────────
# Section 0 - Constants
# ────────────────────────────────────────────────
M2Ft   = 3.28084
Ntolb  = 4.44822

RESULTS_DIR = "PSO_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ────────────────────────────────────────────────
# Helper: unpack PSO particle → full geometry arrays
# ────────────────────────────────────────────────
def unpack_particle(xi, Nominal_Spans, Nominal_Roots, Nominal_Tips,
                    Nominal_Sweeps, MIN_rootcs, MIN_tipcs, MAX_sweeps):
    """Return clipped spans, root_chords, tip_chords, sweeps from a PSO vector."""
    spans       = Nominal_Spans.copy()
    root_chords = Nominal_Roots.copy()
    tip_chords  = Nominal_Tips.copy()
    sweeps      = Nominal_Sweeps.copy()

    spans[2]       *= xi[0]
    root_chords[2] *= xi[1]
    tip_chords[2]  *= xi[2]
    sweeps[0]       = xi[3]
    sweeps[2]       = xi[4]

    spans       = np.clip(spans,       1.0,  40.0)
    root_chords = np.maximum(root_chords, MIN_rootcs)
    tip_chords  = np.maximum(tip_chords,  MIN_tipcs)
    sweeps      = np.clip(sweeps,      10.0, MAX_sweeps)

    return spans, root_chords, tip_chords, sweeps


# ────────────────────────────────────────────────
# Helper: save current best VSP file + geometry JSON
# ────────────────────────────────────────────────
def save_best(iteration, ld, spans, root_chords, tip_chords, sweeps, dihedrals):
    """
    Writes:
      PSO_results/best_vsp_iter{N}_LD{X.XX}.vsp3  — the VSP geometry file
      PSO_results/best_geometry.json               — always-overwritten JSON with
                                                     full geometry + L/D so you can
                                                     reconstruct without the VSP file
    """
    # --- VSP file ---
    vsp_name = os.path.join(RESULTS_DIR,
                            f"best_vsp_iter{iteration:03d}_LD{ld:.4f}.vsp3")
    vsp.WriteVSPFile(vsp_name)
    print(f"  [SAVED] New best VSP → {vsp_name}")

    # --- JSON with all geometry ---
    geom = {
        "iteration":    iteration,
        "L_D_cruise":   ld,
        "spans":        spans.tolist(),
        "root_chords":  root_chords.tolist(),
        "tip_chords":   tip_chords.tolist(),
        "sweeps":       sweeps.tolist(),
        "dihedrals":    dihedrals.tolist(),
    }
    json_path = os.path.join(RESULTS_DIR, "best_geometry.json")
    with open(json_path, "w") as f:
        json.dump(geom, f, indent=2)
    print(f"  [SAVED] Geometry JSON → {json_path}")

    return vsp_name


# ────────────────────────────────────────────────
# analyze_design  (unchanged from original)
# ────────────────────────────────────────────────
def analyze_design(spans, root_chords, tip_chords, sweeps, dihedrals,
                   range_m, mach, payload_N, fuel_frac, tsfc, altitude):

    vsp.ClearVSPModel()
    vsp.ReadVSPFile('SeniorDesign.vsp3')
    wing_id        = vsp.FindGeomsWithName("BodyandWing")[0]
    vstabalizer_id = vsp.FindGeomsWithName("VStabalizer")[0]

    for i in range(len(spans)):
        vsp.SetParmVal(wing_id, "Span",       f"XSec_{i+1}", spans[i])
        vsp.SetParmVal(wing_id, "Root_Chord", f"XSec_{i+1}", root_chords[i])
        vsp.SetParmVal(wing_id, "Tip_Chord",  f"XSec_{i+1}", tip_chords[i])
        vsp.SetParmVal(wing_id, "Sweep",      f"XSec_{i+1}", sweeps[i])
        vsp.SetParmVal(wing_id, "Dihedral",   f"XSec_{i+1}", dihedrals[i])
        vsp.Update()

    basefilename = "PSO_temp_run"
    vspfilename  = f"{basefilename}.vsp3"
    vsp.WriteVSPFile(vspfilename)

    Sref, Swet, WS_areas, VStabalizer_area, MAC, AR = GrabParams.sizing(
        vspfilename, wing_id, vstabalizer_id, len(spans))
    CD0 = GrabParams.parasite(vspfilename, altitude, mach)
    WS_trs = np.array([tip / root for root, tip in zip(root_chords, tip_chords)])

    b                    = sum(spans) * 2
    Wing_wetted_Area     = Swet/Sref * (WS_areas[2]+WS_areas[3]) * (M2Ft**2)
    Fuselage_wetted_Area = Swet/Sref * (WS_areas[0]+WS_areas[1]) * (M2Ft**2)
    Wing_sweep           = sweeps[2]
    Wing_taper           = WS_trs[2]
    Vt_area              = 2 * VStabalizer_area * (M2Ft**2)
    Length_tail          = .55 * root_chords[0] * M2Ft
    Length_fuselage      = root_chords[0] * M2Ft
    Diameter_fuselage    = (spans[0] + 1.15) * 2 * M2Ft
    Payload_lb           = payload_N / Ntolb

    Weight_Distributions = Weights.estimate_aircraft_weights(
        plot_pie=False, export_csv=False,
        Sw=Wing_wetted_Area, AR=AR, lambda_outer_deg=Wing_sweep, taper=Wing_taper,
        V_mach=mach, Svt=Vt_area, Sf=Fuselage_wetted_Area, Lt_ft=Length_tail,
        L_ft=Length_fuselage, D_ft=Diameter_fuselage,
        payload_lb=Payload_lb, fuel_fraction=fuel_frac)

    rho  = compute_density.compute(altitude)
    MTOW = Weight_Distributions['weights_N']['total']

    LD_cruise, CL_cruise, Alpha_cruise, CDi_cruise, CDw_cruise, CD_total_cruise = \
        Aero_Driver.bwb_cruise_analysis(
            False, False, AR, b, Diameter_fuselage, sweeps, root_chords,
            Swet/Sref, WS_areas, mach, CD0, rho, MTOW)

    Offsets    = np.zeros(len(spans))
    Offsets[0] = 0.0
    for i in range(1, len(spans)):
        Offsets[i] = Offsets[i-1] + spans[i-1] * np.tan(np.radians(sweeps[i-1]))

    CG = CGNPSM.compute_cg(
        Weight_Distributions['percentages'], WS_trs, root_chords, spans,
        sweeps, WS_areas*2, Offsets, MAC, mach, LD_cruise, range_m, tsfc)

    AVL.generate_avl_file(
        spans, root_chords, tip_chords, sweeps, dihedrals,
        f'{basefilename}.avl', basefilename, mach, Sref, MAC, b, CG[0])
    NP = AVL.get_neutral_point_from_avl(
        base_name=basefilename, alpha_cruise=Alpha_cruise,
        avl_executable=r"C:\Users\ajbur\OneDrive\Desktop\School\MAE FILES\BURNS\MAE 155A\BWB_Python\AVLFunctions\avl352.exe",
        timeout_seconds=15)
    SM = 100 * (NP - CG) / MAC

    Cost_per_hr = Cost.compute_per_hour_cost(False, tsfc, LD_cruise, MTOW)

    for ext in [".txt", ".avl", "_CompGeom.csv", "_CompGeom.txt", "_ParasiteBuildUp.csv"]:
        fp = f"{basefilename}{ext}"
        if os.path.exists(fp):
            try:    os.remove(fp)
            except PermissionError: pass

    return {
        "MTOW":           MTOW,
        "Static_Margin":  SM,
        "L_D_Cruise":     LD_cruise,
        "Span_Total":     sum(spans) * 2,
        "Fuselage Chord": root_chords[0],
        "Cargo Chord":    root_chords[1],
        "Wing Chord":     root_chords[2],
        "Sweep_inner":    sweeps[0],
        "Sweep_outer":    sweeps[2],
        "Cost_Per_Hour":  Cost_per_hr,
    }


# ────────────────────────────────────────────────
# Acquisition function
# ────────────────────────────────────────────────
def expected_improvement(mu, sigma, best_f, xi=0.01):
    imp = best_f - mu - xi
    Z   = imp / sigma
    ei  = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei


# ────────────────────────────────────────────────
# True (expensive) objective
# ────────────────────────────────────────────────
def true_objective(x, gbest_cost, gbest_pos,
                   Nominal_Spans, Nominal_Roots, Nominal_Tips,
                   Nominal_Sweeps, Nominal_Dihedrals,
                   Nominal_Range, Nominal_Mach, Nominal_Payload,
                   Nominal_Fuel_Frac, Nominal_TSFC, Nominal_Altitude,
                   MIN_rootcs, MIN_tipcs, MAX_sweeps,
                   iteration):
    """
    Evaluates the true L/D for each candidate particle.
    Saves the VSP file and geometry JSON whenever a new global best is found.
    Returns (costs, updated_gbest_cost, updated_gbest_pos).
    """
    n_selected = x.shape[0]
    costs      = np.zeros(n_selected)

    for i in range(n_selected):
        spans, root_chords, tip_chords, sweeps = unpack_particle(
            x[i], Nominal_Spans, Nominal_Roots, Nominal_Tips,
            Nominal_Sweeps, MIN_rootcs, MIN_tipcs, MAX_sweeps)

        try:
            res = analyze_design(
                spans=spans, root_chords=root_chords,
                tip_chords=tip_chords, sweeps=sweeps,
                dihedrals=Nominal_Dihedrals,
                range_m=Nominal_Range, mach=Nominal_Mach,
                payload_N=Nominal_Payload, fuel_frac=Nominal_Fuel_Frac,
                tsfc=Nominal_TSFC, altitude=Nominal_Altitude)

            ld = float(res.get("L_D_Cruise", np.nan))
            costs[i] = -ld if np.isfinite(ld) else 1e6
            print(f"  Eval {i+1}/{n_selected} → L/D = {ld:.4f}   "
                  f"MTOW = {res.get('MTOW', np.nan)/1000:.1f} kN")

            # ── NEW: save VSP + JSON whenever a new global best is found ──
            if costs[i] < gbest_cost:
                gbest_cost = costs[i]
                gbest_pos  = x[i].copy()
                save_best(iteration, ld, spans, root_chords,
                          tip_chords, sweeps, Nominal_Dihedrals)

        except Exception as e:
            print(f"  Eval {i+1}/{n_selected} failed: {e}")
            costs[i] = 1e6

    return costs, gbest_cost, gbest_pos


# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
if __name__ == '__main__':

    # ── Nominal geometry & mission ──────────────────────────────────────────
    Nominal_Spans     = np.array([4.08,  6.50,  19.00, 2.00])
    Nominal_Sweeps    = np.array([62.00, 67.00, 37.00, 40.00])
    Nominal_Roots     = np.array([43.00, 31.18,  9.00, 3.00])
    Nominal_Tips      = np.array([31.18,  9.00,  3.00, 0.80])
    Nominal_Dihedrals = np.array([ 0.00,  0.00,  8.00, 9.25])

    Nominal_Range    = 7000 * 1852
    Nominal_Mach     = 0.85
    Nominal_Altitude = 40000
    Nominal_Payload  = 392000
    Nominal_Fuel_Frac= 0.36
    Nominal_TSFC     = 1.415e-5

    MIN_rootcs = np.array([43.0, 31.18, 0.0, 0.0])
    MIN_tipcs  = np.array([31.18,  0.0, 0.0, 0.0])
    MAX_sweeps = np.array([65.0,  65.0, 45.0, 45.0])

    # ── Optimization variables & bounds ────────────────────────────────────
    bounds_lower = np.array([0.80, 0.90, 0.80, 40.0, 25.0])
    bounds_upper = np.array([1.35, 1.40, 1.50, 70.0, 45.0])
    var_names    = ["Outer span scale", "Outer root chord scale",
                    "Outer tip chord scale", "Inner sweep (deg)", "Outer sweep (deg)"]

    print("Optimization bounds:")
    for name, lo, hi in zip(var_names, bounds_lower, bounds_upper):
        print(f"  {name} : {lo} – {hi}")

    # ── Hyperparameters ─────────────────────────────────────────────────────
    n_particles         = 200
    true_evals_per_iter = 5
    n_iterations        = 15
    min_data_for_gp     = 15
    ei_xi               = 0.01

    # ── History ─────────────────────────────────────────────────────────────
    X_history = np.empty((0, len(bounds_lower)))
    y_history = np.empty((0,))

    # ── Swarm initialisation ────────────────────────────────────────────────
    swarm_pos  = np.random.uniform(bounds_lower, bounds_upper,
                                   size=(n_particles, len(bounds_lower)))
    swarm_vel  = np.random.uniform(-0.5, 0.5, (n_particles, len(bounds_lower)))
    pbest_pos  = swarm_pos.copy()
    pbest_cost = np.full(n_particles, np.inf)
    gbest_pos  = None
    gbest_cost = np.inf

    # ── CSV log (appended each iteration) ───────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "iteration_log.csv")
    log_cols = (["iteration", "L_D_best", "gbest_cost"] +
                var_names +
                [f"span_{i}"        for i in range(len(Nominal_Spans))] +
                [f"root_chord_{i}"  for i in range(len(Nominal_Roots))] +
                [f"tip_chord_{i}"   for i in range(len(Nominal_Tips))]  +
                [f"sweep_{i}"       for i in range(len(Nominal_Sweeps))]+
                [f"dihedral_{i}"    for i in range(len(Nominal_Dihedrals))])
    pd.DataFrame(columns=log_cols).to_csv(log_path, index=False)

    print(f"\nStarting GP+EI PSO  |  particles={n_particles}  "
          f"true_evals/iter={true_evals_per_iter}  iterations={n_iterations}")
    print(f"Results will be saved to: {os.path.abspath(RESULTS_DIR)}\n")

    # ────────────────────────────────────────────────
    # Main loop
    # ────────────────────────────────────────────────
    for it in range(n_iterations):

        print(f"\n Iter {it+1:3d}/{n_iterations}  |  "
          f"True evals this iter: {true_evals_per_iter}  |  "
          f"Best L/D so far: {-gbest_cost if gbest_cost < np.inf else float('nan')}  |  "
          f"Database size: {len(y_history)} \n")

        # ── Surrogate ───────────────────────────────────────────────────────
        if len(X_history) >= min_data_for_gp:
            kernel = (C(1.0, (1e-3, 1e6))
                      * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e3))
                      + WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-15, 1e-1)))
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=10,
                normalize_y=True, random_state=it + 42)
            gp.fit(X_history, y_history)
            mu, sigma = gp.predict(swarm_pos, return_std=True)
            acq = (expected_improvement(mu, sigma, gbest_cost, xi=ei_xi)
                   if gbest_cost < np.inf else -mu)
            top_idx = np.argsort(-acq)[:true_evals_per_iter]
        else:
            if len(X_history) > 0:
                tmp_gp = GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=3)
                tmp_gp.fit(X_history, y_history)
                mu, _ = tmp_gp.predict(swarm_pos, return_std=True)
                top_idx = np.argsort(mu)[:true_evals_per_iter]
            else:
                top_idx = np.random.choice(n_particles, true_evals_per_iter, replace=False)

        candidates = swarm_pos[top_idx]

        # ── True evaluations (with in-loop best saving) ──────────────────
        true_costs, gbest_cost, gbest_pos = true_objective(
            candidates, gbest_cost, gbest_pos,
            Nominal_Spans, Nominal_Roots, Nominal_Tips,
            Nominal_Sweeps, Nominal_Dihedrals,
            Nominal_Range, Nominal_Mach, Nominal_Payload,
            Nominal_Fuel_Frac, Nominal_TSFC, Nominal_Altitude,
            MIN_rootcs, MIN_tipcs, MAX_sweeps,
            iteration=it + 1)

        # ── Update history ───────────────────────────────────────────────
        X_history = (candidates if len(X_history) == 0
                     else np.vstack([X_history, candidates]))
        y_history = (true_costs if len(y_history) == 0
                     else np.concatenate([y_history, true_costs]))

        # ── Personal bests ───────────────────────────────────────────────
        for i, orig_idx in enumerate(top_idx):
            if true_costs[i] < pbest_cost[orig_idx]:
                pbest_cost[orig_idx] = true_costs[i]
                pbest_pos[orig_idx]  = swarm_pos[orig_idx].copy()

        # ── Velocity & position update ───────────────────────────────────
        w         = 0.75 - 0.45 * (it / n_iterations)
        r1        = np.random.rand(n_particles, len(bounds_lower))
        r2        = np.random.rand(n_particles, len(bounds_lower))
        swarm_vel = (w * swarm_vel
                     + 2.05 * r1 * (pbest_pos - swarm_pos)
                     + 2.05 * r2 * (gbest_pos - swarm_pos))
        swarm_pos = np.clip(swarm_pos + swarm_vel, bounds_lower, bounds_upper)

        # ── Log this iteration to CSV ────────────────────────────────────
        if gbest_pos is not None:
            best_spans, best_rc, best_tc, best_sw = unpack_particle(
                gbest_pos, Nominal_Spans, Nominal_Roots, Nominal_Tips,
                Nominal_Sweeps, MIN_rootcs, MIN_tipcs, MAX_sweeps)
            row = ([it+1, -gbest_cost, gbest_cost]
                   + gbest_pos.tolist()
                   + best_spans.tolist()
                   + best_rc.tolist()
                   + best_tc.tolist()
                   + best_sw.tolist()
                   + Nominal_Dihedrals.tolist())
            pd.DataFrame([row], columns=log_cols).to_csv(
                log_path, mode='a', header=False, index=False)

    # ────────────────────────────────────────────────
    # Final results
    # ────────────────────────────────────────────────
    best_LD = -gbest_cost
    best_spans, best_rc, best_tc, best_sw = unpack_particle(
        gbest_pos, Nominal_Spans, Nominal_Roots, Nominal_Tips,
        Nominal_Sweeps, MIN_rootcs, MIN_tipcs, MAX_sweeps)

    print("\n" + "═"*80)
    print("GP + EI ASSISTED PSO FINISHED")
    print(f"Best Cruise L/D : {best_LD:.4f}")
    print()

    # ── Full geometry to reconstruct without VSP ─────────────────────────
    print("Full geometry to recreate best design:")
    print(f"  spans       = {best_spans.tolist()}")
    print(f"  root_chords = {best_rc.tolist()}")
    print(f"  tip_chords  = {best_tc.tolist()}")
    print(f"  sweeps      = {best_sw.tolist()}")
    print(f"  dihedrals   = {Nominal_Dihedrals.tolist()}")
    print()
    print("PSO scaled variables at optimum:")
    for name, val in zip(var_names, gbest_pos):
        print(f"  {name} = {val:.6f}")

    # ── Re-evaluate with full metrics ────────────────────────────────────
    print("\nRe-evaluating best design with full metrics...")
    final_result = analyze_design(
        spans=best_spans, root_chords=best_rc, tip_chords=best_tc,
        sweeps=best_sw, dihedrals=Nominal_Dihedrals,
        range_m=Nominal_Range, mach=Nominal_Mach,
        payload_N=Nominal_Payload, fuel_frac=Nominal_Fuel_Frac,
        tsfc=Nominal_TSFC, altitude=Nominal_Altitude)

    print("\nFinal metrics:")
    for key, value in final_result.items():
        print(f"  {key} = {value}")

    # ── Save final geometry JSON (overwrites the last best-in-loop save) ─
    save_best(n_iterations, best_LD, best_spans, best_rc,
              best_tc, best_sw, Nominal_Dihedrals)

    print(f"\nAll outputs saved to: {os.path.abspath(RESULTS_DIR)}/")
    print(f"  best_geometry.json    — full geometry + L/D")
    print(f"  best_vsp_iter***.vsp3 — VSP file at each new best")
    print(f"  iteration_log.csv     — full history of every iteration")