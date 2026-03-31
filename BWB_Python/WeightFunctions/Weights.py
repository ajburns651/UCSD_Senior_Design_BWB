## === Script To Compute Weight Fractions === ##
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict
import csv
import os


@dataclass
class WeightBreakdown:
    wing: float
    vertical_tail: float
    fuselage: float
    main_landing_gear: float
    nose_landing_gear: float
    installed_engine: float
    fuel_system: float
    flight_controls: float
    hydraulics: float
    avionics: float
    electrical: float
    air_conditioning_anti_ice: float
    furnishings: float
    fuel: float
    payload: float
    total: float


def estimate_aircraft_weights(
    # VALUES OVERRIDED BY MASTER SCRIPT
    Sw: float = 6069.12,
    AR: float = 4.92347,
    lambda_outer_deg: float = 37.0,
    taper: float = 0.3333,
    V_mach: float = 0.85,
    Svt: float = 21.554,
    Sf: float = 6314.75,
    Lt_ft: float = 25 * 3.28084,
    L_ft: float = 39.221 * 3.28084,
    D_ft: float = 10.7132 * 3.28084,
    payload_lb: float = 392000 / 4.44822,
    fuel_fraction: float = 0.36,

    # Airfoil Dependent
    tc: float = 0.06853,

    # Other
    Wdg_guess_lb: float = 10000,
    Nz: float = 3.8,
    Vpr_ft3: float = 72530,
    Pdelta_psi: float = 8.0,
    Nl_factor: float = 1.5,
    Lm_in: float = 20 * 12,
    Ln_in: float = 10,
    Wen_lb: float = 60240.6 * 0.224809,
    Nen: int = 2,
    Vt_gal: float = 40000,
    fuel_in_wings_factor: float = 1.0,
    Wuav_lb: float = 1100,
    Np: int = 250 + 13,
    max_iter: int = 100,
    tol: float = 1.0,
    plot_pie: bool = True,
    small_slice_threshold: float = 2.0,
    export_csv: bool = True,
    csv_filename: str = "aircraft_weight_breakdown.csv"
) -> Dict:
    g = 32.174
    rho_lb_ft3 = 0.01889
    rho_slug_ft3 = rho_lb_ft3 / g

    V_fps = V_mach * 295 * 3.28084
    q = 0.5 * rho_slug_ft3 * V_fps**2

    lambda_outer = math.radians(lambda_outer_deg)
    M = V_mach

    W_press = 11.9 * (Vpr_ft3 * Pdelta_psi)**0.271
    Vi = Vt_gal * 0.9
    Nt = 3
    Kh = 0.12

    Wprior = Wdg_guess_lb
    converged = False

    for i in range(1, max_iter + 1):
        Wl = Wprior
        Nl = Nz * Nl_factor

        W_wing = 0.036 * Sw**0.758 * fuel_in_wings_factor**0.0035 * \
                 (AR / math.cos(lambda_outer)**2)**0.6 * \
                 q**0.006 * taper**0.04 * \
                 (100 * tc / math.cos(lambda_outer))**(-0.3) * \
                 (Nz * Wprior)**0.49

        W_vt = 0.073 * (Nz * Wprior)**0.376 * q**0.122 * Svt**0.873

        W_fuselage = 0.052 * Sf**1.086 * (Nz * Wprior)**0.177 * \
                     Lt_ft**(-0.051) * (L_ft / D_ft)**(-0.072) * q**0.241 + W_press

        W_main_gear = 0.095 * (Nl_factor * Nz * Wl)**0.768 * (Lm_in / 12)**0.409
        W_nose_gear = 0.125 * (Nl_factor * Nz * Wl)**0.566 * (Ln_in / 12)**0.845

        W_installed_engine = 2.575 * Wen_lb**0.922 * Nen

        W_fuel_sys = 2.49 * Vt_gal**0.726 * (1 / (1 + Vi / Vt_gal))**0.363 * \
                     Nt**0.242 * Nen**0.157

        W_flight_controls = 0.053 * L_ft**1.536 * (63.13*3.28084)**0.371 * \
                            (Nz * Wprior * 1e-4)**0.80

        W_hydraulics = Kh * Wprior**0.8 * M**0.5

        W_avionics = 2.117 * Wuav_lb**0.933

        W_electrical = 12.57 * (W_fuel_sys + W_avionics)**0.51

        W_ac_anti_ice = 0.265 * Wprior**0.52 * Np**0.68 * \
                        W_avionics**0.17 * M**0.08

        W_furnishings = 0.0582 * Wprior - 65

        W_empty = (W_wing + W_vt + W_fuselage + W_main_gear + W_nose_gear +
                   W_installed_engine + W_fuel_sys + W_flight_controls +
                   W_hydraulics + W_avionics + W_electrical +
                   W_ac_anti_ice + W_furnishings)

        err = W_empty - Wprior

        if abs(err) < tol:
            converged = True
            break

        Wprior = W_empty

    if not converged:
        print(f"Warning: did NOT converge within {max_iter} iterations (error = {err:.1f} lb)")

    WTO = (W_empty + payload_lb)/(1-fuel_fraction)
    W_fuel = fuel_fraction * WTO

    weights_lb = WeightBreakdown(
        wing=W_wing,
        vertical_tail=W_vt,
        fuselage=W_fuselage,
        main_landing_gear=W_main_gear,
        nose_landing_gear=W_nose_gear,
        installed_engine=W_installed_engine,
        fuel_system=W_fuel_sys,
        flight_controls=W_flight_controls,
        hydraulics=W_hydraulics,
        avionics=W_avionics,
        electrical=W_electrical,
        air_conditioning_anti_ice=W_ac_anti_ice,
        furnishings=W_furnishings,
        fuel=W_fuel,
        payload=payload_lb,
        total=WTO
    )

    lb_to_kg = 0.45359237
    lb_to_N = 4.44822
    weights_kg = {k: round(v * lb_to_kg, 3) for k, v in weights_lb.__dict__.items()}
    weights_N = {k: round(v * lb_to_N, 3) for k, v in weights_lb.__dict__.items()}

    percentages = {k: round(100 * v / WTO, 3) for k, v in weights_lb.__dict__.items()}

    component_names = [
            "Wing", "Vertical Tail", "Fuselage", "Main Landing Gear", "Nose Landing Gear",
            "Installed Engine", "Fuel System", "Flight Controls", "Hydraulics",
            "Avionics", "Electrical", "Air Conditioning & Anti-Ice", "Furnishings",
            "Fuel", "Payload", "Total"
        ]
    
    attr_names = [
            "wing", "vertical_tail", "fuselage", "main_landing_gear", "nose_landing_gear",
            "installed_engine", "fuel_system", "flight_controls", "hydraulics",
            "avionics", "electrical", "air_conditioning_anti_ice", "furnishings",
            "fuel", "payload"
        ]

    result = {
        "weights_N": weights_N,
        "weights_kg": weights_kg,
        "percentages": percentages,
        "labels": component_names
    }

    # ────────────────────────────────────────────────
    # CSV EXPORT
    # ────────────────────────────────────────────────
    if export_csv:

        row_N  = [weights_N[attr]  for attr in attr_names] + [weights_N["total"]]
        row_kg = [weights_kg[attr] for attr in attr_names] + [weights_kg["total"]]
        row_pct = [percentages[attr] for attr in attr_names] + [100.0]

        filename = csv_filename
        if os.path.exists(filename):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(filename):
                filename = f"{base}_{counter}{ext}"
                counter += 1

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            writer.writerow(["Aircraft Weight Breakdown Export"])
            writer.writerow([])

            writer.writerow(["Component", "Weight (N)", "Percentage (%)"])
            for name, val, pct in zip(component_names, row_N, row_pct):
                writer.writerow([name, f"{val:,.1f}", f"{pct:.3f}"])

            writer.writerow([])
            writer.writerow([])

            writer.writerow(["Component", "Weight (kg)", "Percentage (%)"])
            for name, val, pct in zip(component_names, row_kg, row_pct):
                writer.writerow([name, f"{val:,.1f}", f"{pct:.3f}"])

            writer.writerow([])
            writer.writerow(["Notes:"])
            writer.writerow(["MTOW = Maximum Takeoff Weight"])
            writer.writerow([f"Converged: {converged}, Iterations: {i}"])
            writer.writerow([f"Operating Empty Weight = {WTO*lb_to_N:,.0f} N"])

        print(f"→ Exported to {filename}")

    # ────────────────────────────────────────────────
    # PIE CHART
    # ────────────────────────────────────────────────
    if plot_pie:
        # 15 visually distinct colors — same vibrant palette style as the DOC cost chart,
        # no two colors are repeated or closely similar
        palette = [
            "#E63946",  # vivid red        → Wing
            "#F77F00",  # orange           → Vertical Tail
            "#F4D35E",  # golden yellow    → Fuselage
            "#3BB273",  # forest green     → Main Landing Gear
            "#4CC9F0",  # sky blue         → Nose Landing Gear
            "#1D3557",  # navy             → Installed Engine
            "#9B5DE5",  # purple           → Fuel System
            "#06D6A0",  # emerald teal     → Flight Controls
            "#EF476F",  # hot pink         → Hydraulics
            "#118AB2",  # ocean blue       → Avionics
            "#FFB703",  # amber            → Electrical
            "#6D6875",  # dusty mauve      → Air Conditioning & Anti-Ice
            "#C77DFF",  # lavender         → Furnishings
            "#80B918",  # lime green       → Fuel
            "#073B4C",  # deep teal        → Payload
        ]

        labels = component_names[:-1]  # exclude Total
        percents = np.array([percentages[attr] for attr in attr_names])
        colors   = np.array(palette)

        small_thresh = small_slice_threshold
        small_idx = percents < small_thresh
        large_idx = ~small_idx

        small_percents = percents[small_idx]
        small_labels   = np.array(labels)[small_idx]
        small_colors   = colors[small_idx]
        large_percents = percents[large_idx]
        large_labels   = np.array(labels)[large_idx]
        large_colors   = colors[large_idx]

        # Evenly distribute small slices among large ones (translated from MATLAB)
        new_percents = []
        new_labels   = []
        new_colors   = []
        Ls = len(large_percents)
        Ss = len(small_percents)
        if Ss > 0:
            slot_positions = np.round(np.linspace(1, Ls + 1, Ss)).astype(int)
        else:
            slot_positions = np.array([])
        slot_counter = 0
        for ii in range(Ls):
            new_percents.append(large_percents[ii])
            new_labels.append(large_labels[ii])
            new_colors.append(large_colors[ii])
            if slot_counter < Ss and slot_positions[slot_counter] == ii + 1:
                new_percents.append(small_percents[slot_counter])
                new_labels.append(small_labels[slot_counter])
                new_colors.append(small_colors[slot_counter])
                slot_counter += 1
        while slot_counter < Ss:
            new_percents.append(small_percents[slot_counter])
            new_labels.append(small_labels[slot_counter])
            new_colors.append(small_colors[slot_counter])
            slot_counter += 1

        new_percents = np.array(new_percents)
        new_labels   = np.array(new_labels)
        new_colors   = np.array(new_colors)

        # Explode only small slices
        explode = np.zeros(len(new_percents))
        explode[new_percents < small_thresh] = 0.1

        fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
        wedges, texts, autotexts = ax.pie(
            new_percents,
            explode=explode,
            labels=None,
            colors=new_colors,
            autopct='%1.1f%%',
            shadow=False,
            startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(linewidth=1.2, edgecolor="white"),
        )

        # Custom label formatting & positioning
        for j, (pct, txt) in enumerate(zip(new_percents, autotexts)):
            txt.set_fontsize(10)
            txt.set_color('white')
            txt.set_horizontalalignment('center')

            x, y = txt.get_position()
            norm = math.sqrt(x**2 + y**2)
            if norm < 1e-6:
                dir_x, dir_y = 1.0, 0.0
            else:
                dir_x, dir_y = x / norm, y / norm

            if pct >= small_thresh:
                new_r = 0.62
                txt.set_position((dir_x * new_r, dir_y * new_r))
                txt.set_fontweight('bold')
            else:
                outer_r = 1.2
                if abs(dir_y) > 0.9:
                    outer_r = 1.12
                txt.set_position((dir_x * outer_r, dir_y * outer_r))
                txt.set_color('black')  # small-slice labels outside the pie → dark text

        # Legend with weight values
        legend_labels = [
            f"{lbl}  ({pct:.1f}%)"
            for lbl, pct in zip(new_labels, new_percents)
        ]
        ax.legend(
            wedges, legend_labels,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            fontsize=9,
            framealpha=0.95,
            edgecolor="#cccccc",
        )

        ax.set_title(
            f"Aircraft Weight Breakdown\nMTOW ≈ {weights_N['total']:,.0f} N  ({weights_kg['total']:,.0f} kg)",
            fontsize=14,
            fontweight="bold",
            pad=16,
        )
        plt.tight_layout()
        plt.show()

    return result


# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────
if __name__ == "__main__":
    result = estimate_aircraft_weights(
        Wdg_guess_lb=15000,
        plot_pie=False,
        export_csv=False
    )

    print(f"\nWeights (N) = {{ {', '.join(f'{k}: {v:.0f}N' for k,v in result['weights_N'].items())} }}")