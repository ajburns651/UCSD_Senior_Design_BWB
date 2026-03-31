import matplotlib.pyplot as plt
import numpy as np

def compute_per_hour_cost(flag_plot, TSFC, L_D, W,
                                seats=250,
                                range_nm=7246,
                                M=0.85,
                                aircraft_price=210e6,
                                engine_price=25e6,
                                Ne=2):
    """
    Computes Direct Operating Cost per seat-mile for a transport aircraft.
    Combines Raymer maintenance model (MATLAB) with per-seat-mile breakdown (Python).

    Inputs:
        flag_plot       : bool — show pie chart
        TSFC            : kg/(N·s)
        L_D             : cruise lift-to-drag ratio
        W               : aircraft cruise weight (N)
        seats           : passenger capacity
        range_nm        : mission range (nmi)
        M               : cruise Mach number
        aircraft_price  : total aircraft price ($)
        engine_price    : price per engine ($)
        Ne              : number of engines

    Returns:
        dict with per-hour and per-seat-mile costs
    """

    # ────────────────────────────────────────────────
    # Mission Parameters
    # ────────────────────────────────────────────────
    a_knots       = 573                        # speed of sound @ 40,000 ft
    V_knots       = M * a_knots
    V_mph         = V_knots * 1.15078
    flight_time_hr = range_nm / V_knots        # hours

    seat_miles_hr  = seats * V_mph             # seat-miles per hour

    # ────────────────────────────────────────────────
    # 1. Fuel Cost
    # ────────────────────────────────────────────────
    fuel_price   = 4.5       # $/gallon (Argus Data)
    rho_jetA     = 0.804     # kg/L
    L_per_gal    = 3.785     # L/gallon
    hedge_factor = 0.975     # fuel hedge discount

    T            = W / L_D                        # thrust (N)
    mdot         = TSFC * T                       # kg/s
    fuel_kg_hr   = mdot * 3600
    fuel_gal_hr  = (fuel_kg_hr / rho_jetA) / L_per_gal
    fuel_cost_hr = fuel_gal_hr * fuel_price * hedge_factor

    # ────────────────────────────────────────────────
    # 2. Crew Cost
    # ────────────────────────────────────────────────
    pilot_cost_hr     = 260000 / 900              # 2 pilots
    total_pilot_hr    = 2 * pilot_cost_hr

    fa_cost_hr        = 80000 / 900               # 8 FAs (long-haul, 1 per 50 pax min)
    total_fa_hr       = 8 * fa_cost_hr

    crew_cost_hr      = total_pilot_hr + total_fa_hr

    # ────────────────────────────────────────────────
    # 3. Maintenance — Raymer Model (Eq. 18.12 / 18.13)
    # ────────────────────────────────────────────────
    Ca_m  = (aircraft_price - Ne * engine_price) / 1e6   # airframe cost ($M)
    Ce_m  = engine_price / 1e6                            # engine cost ($M)

    # Material cost per flight hour (Raymer Eq. 18.12)
    material_cost_FH = 3.3 * Ca_m + 14.2 + (58 * Ce_m - 26.1) * Ne

    # Material cost per cycle (Raymer Eq. 18.13)
    material_cost_cycle   = 4.0 * Ca_m + 9.3 + (7.5 * Ce_m + 5.6) * Ne
    cycles_per_hour       = 1.0 / flight_time_hr           # 1 cycle per flight
    material_cycle_cost_hr = material_cost_cycle * cycles_per_hour

    # Labor cost
    MMH_FH        = 8                                      # maintenance man-hours per FH
    labor_rate    = 60                                     # $/hr per mechanic
    labor_cost_hr = MMH_FH * labor_rate

    maintenance_material_hr = material_cost_FH + material_cycle_cost_hr
    maintenance_cost_hr     = maintenance_material_hr + labor_cost_hr

    # ────────────────────────────────────────────────
    # 4. Fees (Airport + Navigation)
    # ────────────────────────────────────────────────
    airport_fees_total = 10000                             # $ per mission
    nav_fees_total     = 7000                              # $ per mission

    airport_fee_hr = (airport_fees_total / range_nm) * V_knots
    nav_fee_hr     = (nav_fees_total     / range_nm) * V_knots
    fee_cost_hr    = airport_fee_hr + nav_fee_hr

    # ────────────────────────────────────────────────
    # 5. Base → Total (Insurance, Oil, Depreciation via Raymer markup)
    # ────────────────────────────────────────────────
    base_cost_hr         = fuel_cost_hr + crew_cost_hr + maintenance_cost_hr + fee_cost_hr
    total_cost_hr        = base_cost_hr / 0.86             # markup covers insurance+oil+depreciation

    depreciation_cost_hr = 0.09 * total_cost_hr
    insurance_cost_hr    = 0.03 * total_cost_hr
    oil_cost_hr          = 0.02 * total_cost_hr

    # ────────────────────────────────────────────────
    # 6. Per-Seat-Mile Conversion
    # ────────────────────────────────────────────────
    def per_sm(cost_hr):
        return cost_hr / seat_miles_hr

    results = {
        # Per hour
        "fuel_cost_hr":           fuel_cost_hr,
        "crew_cost_hr":           crew_cost_hr,
        "maintenance_cost_hr":    maintenance_cost_hr,
        "fee_cost_hr":            fee_cost_hr,
        "insurance_cost_hr":      insurance_cost_hr,
        "oil_cost_hr":            oil_cost_hr,
        "depreciation_cost_hr":   depreciation_cost_hr,
        "total_cost_hr":          total_cost_hr,
        # Per seat-mile
        "fuel_casm":              per_sm(fuel_cost_hr),
        "pilot_casm":             per_sm(total_pilot_hr),
        "fa_casm":                per_sm(total_fa_hr),
        "maintenance_mat_casm":   per_sm(maintenance_material_hr),
        "maintenance_labor_casm": per_sm(labor_cost_hr),
        "airport_fee_casm":       per_sm(airport_fee_hr),
        "nav_fee_casm":           per_sm(nav_fee_hr),
        "insurance_casm":         per_sm(insurance_cost_hr),
        "oil_casm":               per_sm(oil_cost_hr),
        "depreciation_casm":      per_sm(depreciation_cost_hr),
        "total_casm":             per_sm(total_cost_hr),
    }

    # ────────────────────────────────────────────────
    # 7. Pie Chart
    # ────────────────────────────────────────────────
    if flag_plot:
            casm_values = [
                results["fuel_casm"],
                results["pilot_casm"],
                results["fa_casm"],
                results["maintenance_mat_casm"],
                results["maintenance_labor_casm"],
                results["airport_fee_casm"],
                results["nav_fee_casm"],
                results["insurance_casm"],
                results["oil_casm"],
                results["depreciation_casm"],
            ]
            labels = [
                "Fuel",
                "Pilots",
                "Flight Attendants",
                "Maint. Materials",
                "Maint. Labor",
                "Airport Fees",
                "Navigation Fees",
                "Insurance (3%)",
                "Oil (2%)",
                "Depreciation (9%)",
            ]
            colors = [
                "#E63946", "#F4D35E", "#3BB273", "#1D3557", "#9B5DE5",
                "#F77F00", "#4CC9F0", "#C1121F", "#6D6875", "#06D6A0",
            ]

            total = results["total_casm"]

            # Build legend labels with values
            legend_labels = [
                f"{lbl}  ${v:.4f}  ({v/total*100:.1f}%)"
                for lbl, v in zip(labels, casm_values)
            ]

            fig, ax = plt.subplots(figsize=(12, 8), facecolor="white")

            wedges, texts, autotexts = ax.pie(
            casm_values,
            labels=None,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 4 else "",
            startangle=140,
            pctdistance=0.78,
            wedgeprops=dict(linewidth=1.2, edgecolor="white"),
            shadow=False,
            textprops=dict(fontsize=14),          # ← wedge label size
        )

            for at in autotexts:
                at.set_fontsize(14)                    # ← percentage text on wedges
                at.set_fontweight("bold")
                at.set_color("white")

            ax.legend(
                wedges,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                fontsize=13,                           # ← legend text
                framealpha=0.95,
                edgecolor="#cccccc",
                title="Cost Category",
                title_fontsize=14,                     # ← legend title
            )

            ax.set_title(
                f"Direct Operating Cost Breakdown — Per Seat-Mile\n"
                f"Total CASM = ${total:.4f}   |   DOC = ${results['total_cost_hr']:,.0f} /hr",
                fontsize=15,                           # ← title
                fontweight="bold",
                pad=20,
            )

            plt.tight_layout()
            plt.savefig("DOC_Breakdown.png", dpi=150, bbox_inches="tight")
            plt.show()

    return results