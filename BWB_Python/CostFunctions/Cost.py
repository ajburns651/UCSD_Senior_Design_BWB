# Script to Compute Airplane Cost
import matplotlib.pyplot as plt
import numpy as np

def compute_per_hour_cost(flag_plot,TSFC, L_D, W):
    # ────────────────────────────────────────────────
    # Fuel Cost
    # ────────────────────────────────────────────────
    fuel_price = 4.5          # $/gallon Using Argus Data
    rho_jetA = 0.804        # kg/L
    L_per_gal = 3.785       # liters/gallon

    T = W / L_D             # Thrust (Newtons)
    mdot = TSFC * T         # kg/s
    fuel_kg_hr = mdot * 3600
    fuel_L_hr = fuel_kg_hr / rho_jetA
    fuel_gal_hr = fuel_L_hr / L_per_gal

    fuel_cost_hr = fuel_gal_hr * fuel_price

    #print(f"Fuel cost = ${fuel_cost_hr:,.0f} per flight hour")

    # ────────────────────────────────────────────────
    # Crew Cost
    # ────────────────────────────────────────────────
    # 2 pilots
    pilot_annual_salary = 260000    # $/year total compensation
    pilot_hours_per_year = 900

    # 8 flight attendants
    fa_annual_salary = 80000        # $/year
    fa_hours_per_year = 900

    pilot_cost_hr = pilot_annual_salary / pilot_hours_per_year
    total_pilot_cost_hr = 2 * pilot_cost_hr

    fa_cost_hr = fa_annual_salary / fa_hours_per_year
    total_fa_cost_hr = 8 * fa_cost_hr

    crew_cost_hr = total_pilot_cost_hr + total_fa_cost_hr
    crew_cost_display = round(crew_cost_hr, -2)     # round to nearest 100 like original

    #print(f"Crew cost = ${crew_cost_display:,.0f} per flight hour")

    # ────────────────────────────────────────────────
    # Maintenance Cost
    # ────────────────────────────────────────────────
    engine_maintenance_hr = 2200    # $/hr (both engines)
    airframe_maintenance_hr = 1200  # $/hr
    maintenance_cost_hr = engine_maintenance_hr + airframe_maintenance_hr

    #print(f"Maintenance cost = ${maintenance_cost_hr:,.0f} per flight hour")

    # ────────────────────────────────────────────────
    # Fee cost
    # ────────────────────────────────────────────────
    total_mission_fees = 17000      # $ per LAX-DXB mission
    block_time = 16                 # hours

    fee_cost_hr = total_mission_fees / block_time
    #print(f"Fees cost = ${round(fee_cost_hr):,} per flight hour")

    # ────────────────────────────────────────────────
    # Total Cost
    # ────────────────────────────────────────────────
    total_per_hour_flight_cost = (
        fee_cost_hr +
        maintenance_cost_hr +
        crew_cost_hr +
        fuel_cost_hr
    )

    # Note: your original fprintf used a hardcoded value → here we compute it
    #print(f"Total Cost Per Flight Hour = ${total_per_hour_flight_cost:,.0f}")

    # ────────────────────────────────────────────────
    # Pie Chart
    # ────────────────────────────────────────────────
    if flag_plot == True:
        costs = [fuel_cost_hr, crew_cost_hr, maintenance_cost_hr, fee_cost_hr]
        labels = ['Fuel', 'Crew', 'Maintenance', 'Fees']

        # Optional: show percentage and value in pie slices
        def autopct_func(pct):
            return f'{pct:.1f}%\n${pct/100 * sum(costs):,.0f}'

        plt.figure(figsize=(9, 7))
        plt.pie(costs, labels=labels, autopct=autopct_func,
                startangle=90, shadow=True, explode=[0.05]*4)
        plt.title('Direct Operating Cost Breakdown Per Flight Hour')
        plt.axis('equal')  # equal aspect ratio → true circle

        # Alternative (simpler, like original MATLAB):
        # plt.pie(costs, labels=labels)
        # plt.title('Direct Operating Cost Breakdown Per Flight Hour')
        # plt.legend(loc='best')

        plt.show()
    
    return total_per_hour_flight_cost