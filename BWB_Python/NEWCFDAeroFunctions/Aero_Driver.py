## === Script To Compute and Plot Aero Parameters === ##
import numpy as np
import matplotlib.pyplot as plt

# Main Function
def bwb_cruise_analysis(plotting_flag,printing_flag,AspectRatio,Wingspan,Fuselage_diameter,Sweeps,Root_Chords,Swet_over_Sref,Areas,Mach,Cd0,Density,MTOW):
    
    geom = {}
    aero = {}
    
    # ================= AUTO COMPUTED =================
    ### Auto Computed
    geom['AR']         = AspectRatio
    geom['b']          = Wingspan                   # m
    geom['d']          = Fuselage_diameter          # m (ballpark)
    geom['Sweeps']     = Sweeps                     # deg
    geom['c_ref']      = Root_Chords
    geom['Swet_Sref']  = Swet_over_Sref
    geom['Swr']        = Areas
    Sref = sum(Areas)*2
    geom['Swr_frac']   = geom['Swr'] / np.sum(geom['Swr'])

    aero['M']          = Mach
    V                  = aero['M'] * 295.1
    aero['Cd0']        = Cd0

    rho                = Density      # kg/m³

    # ================= USER INPUT =================
    # Airfoil Dependent
    geom['t_c']        = np.array([0.11, 0.11, 0.07, 0.07])
    
    aero['e']          = 0.925          # Oswald efficiency (from reference)
    aero['ka']         = 0.95  

    aero['Cla']        = 0.0854 * 180/np.pi         # 2D lift slope → per radians    

    aero['Clalpha_sec'] = np.array([aero['Cla'], aero['Cla'], aero['Cla'], aero['Cla']])   # 2D lift slope → per radians  
    aero['Cl0_sec']     = np.array([0.2991, 0.2991, 0.2991, 0.2991])   
    aero['Cd0_sec']     = np.array([0.0126,0.0126,0.0126,0.0126])


    # ================= AERODYNAMIC MODEL =================
    aoa_deg = np.linspace(-5, 10, 40)
    aoa_rad = np.deg2rad(aoa_deg)

    # 1. Lift and and Base Drag (missing wave)
    from AeroFunctions import compute_drag_polar
    CL, CLalpha, CD_base = compute_drag_polar.determine_parameters(aoa_rad, geom, aero)

    # 3. Wave drag
    from AeroFunctions import compute_wave_drag
    CD_wave = compute_wave_drag.determine_wave_drag(CL, geom, aero)

    # 4. Total drag coefficient
    CD = CD_base + CD_wave

    # ================= CRUISE ANALYSIS =================
    q = 0.5 * rho * V**2
    CL_cruise = MTOW / (q * Sref)

    # Find closest point in the computed polar
    idx = np.argmin(np.abs(CL - CL_cruise))
    CD_total_cruise = CD[idx]
    alpha_cruise_deg = aoa_deg[idx]

    # Component drag breakdown at cruise
    k_factor = 1.0 / (np.pi * aero['e'] * geom['AR'])
    CDi_cruise = k_factor * CL_cruise**2
    CDw_cruise = CD_wave[idx]

    # ================= OUTPUTS =================
    if printing_flag==True:
        print("\n" + "="*50)
        print("         FULL BWB AERODYNAMIC BREAKDOWN")
        print("="*50)
        print(f"MTOW                  : {MTOW:,.0f} N")
        print(f"Cruise Mach           : {aero['M']:.2f}")
        print(f"Reference Area (Sref) : {Sref:.3f} m²")
        print("-"*50)
        print(f"Lift Coefficient (CL) : {CL_cruise:.5f}")
        print(f"Angle of Attack (AoA) : {alpha_cruise_deg:.5f} deg")
        print("-"*50)
        print(f"Parasite Drag (CD0)   : {aero['Cd0']:.5f}")
        print(f"Induced Drag (CDi)    : {CDi_cruise:.5f}")
        print(f"Wave Drag (CDw)       : {CDw_cruise:.6f}")
        print(f"Total Drag (CD)       : {CD_total_cruise:.5f}")
        print("-"*50)
        print(f"L/D Ratio             : {CL_cruise / CD_total_cruise:.2f}")
        print(f"Lift Curve Slope (CLa): {CLalpha[0]:.4f} /rad")
        print("="*50 + "\n")

    # ================= PLOTTING =================
    LD_cruise = CL_cruise / CD_total_cruise

    if plotting_flag==True:
        from AeroFunctions import plot_aero_parameters
        plot_aero_parameters.plot_aero(CL, CD, CL_cruise, CD_total_cruise, LD_cruise, geom, aoa_deg)

    return LD_cruise, CL_cruise, alpha_cruise_deg, CDi_cruise, CDw_cruise, CD_total_cruise

if __name__ == "__main__":
    bwb_cruise_analysis()