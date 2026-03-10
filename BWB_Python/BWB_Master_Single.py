# ────────────────────────────────────────────────
# 155A BWB - Single Design Script
# ────────────────────────────────────────────────
import openvsp as vsp
import numpy as np
import matplotlib as plt
import csv
import os
import datetime

# ────────────────────────────────────────────────
# Section 0 - Define Frequently Used Constants
# ────────────────────────────────────────────────
M2Ft = 3.28084
Ntolb = 4.44822

# ────────────────────────────────────────────────
# Section 1 - Define High Level Parameters
# ────────────────────────────────────────────────
Obj = {}

Obj['Range'] = 7000 * 1852        # Destination Range (Nmi -> Meters)
Obj['Mach'] = 0.85                # Mach Number
Obj['Altitude'] = 40000           # Cruise height (Feet)
Obj['Payload Weight'] = 392000    # Payload weight (Newtons)
Obj['Fuel Fraction'] = 0.36       # Fuel Fraction
Obj['TSFC'] = 1.415 * 10**(-5)    # Engine TSFC

# ────────────────────────────────────────────────
# Section 2 - Define Shape Parameters
# ────────────────────────────────────────────────
# User Input
WS_spans = np.array([4.08, 6.50, 19.00, 2.00])          # Span of each section (m) – length per segment
WS_rootcs = np.array([43.00, 31.18, 9.00, 3.00])        # Root chord of each section (m)
WS_tipcs  = np.array([31.18, 9.00, 3.00, 0.80])         # Tip chord of each section (m)
WS_sweeps = np.array([62.00, 67.00, 37.00, 40.00])      # LE sweep angle of each section (deg)
WS_dihedrals = np.array([0.00, 0.00, 8.00, 9.25])       # Dihedral angle of each section (deg)

# Derived
N_WingSections = len(WS_spans)

# ────────────────────────────────────────────────
# Section 3 - Define Airfoil Parameters (Currently Defined In Scripts)
# ────────────────────────────────────────────────
Aero = {}

# ────────────────────────────────────────────────
# Section 4 - Update OpenVSP Geometry To Shape Values 
# ────────────────────────────────────────────────
vsp.ReadVSPFile('SeniorDesign.vsp3')

# Locate the wing geometry id values
wing_id = vsp.FindGeomsWithName("BodyandWing")[0]
vstabalizer_id = vsp.FindGeomsWithName("VStabalizer")[0]

# Set parameters for Each Wing Section ("XSec_i" group)
for i in range(N_WingSections):
    vsp.SetDriverGroup(wing_id, i+1, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
    vsp.SetParmVal(wing_id, "Span", f"XSec_{i+1}", WS_spans[i])
    vsp.SetParmVal(wing_id, "Root_Chord", f"XSec_{i+1}", WS_rootcs[i])
    vsp.SetParmVal(wing_id, "Tip_Chord", f"XSec_{i+1}", WS_tipcs[i])
    vsp.SetParmVal(wing_id, "Sweep", f"XSec_{i+1}", WS_sweeps[i])
    vsp.SetParmVal(wing_id, "Dihedral", f"XSec_{i+1}", WS_dihedrals[i])
    vsp.Update() # Update the model after setting section properties (important for connected sections)

# Save the geometry file
basefilename = f"BWB_Single_Run"
vspfilename = f"{basefilename}.vsp3"
VSPFILE = vsp.WriteVSPFile(vspfilename)

# ────────────────────────────────────────────────
# Section 5 - Compute Related OpenVSP Values (Areas, Taper Ratios, CD0, NP, etc)
# ────────────────────────────────────────────────
from OpenVSPHooks import GrabParams
Sref, Swet, WS_areas, VStabalizer_area, MAC, AR = GrabParams.sizing(vspfilename, wing_id, vstabalizer_id, N_WingSections)
CD0 = GrabParams.parasite(vspfilename, Obj['Altitude'], Obj['Mach'])
WS_trs = np.array([tip / root for root, tip in zip(WS_rootcs, WS_tipcs)]) # Compute Taper Ratios

# ────────────────────────────────────────────────
# Section 6 - Compute Weights
# ────────────────────────────────────────────────
from WeightFunctions import Weights
# Compute Inputs
b = sum(WS_spans)*2
Wing_wetted_Area = Swet/Sref * (WS_areas[2] + WS_areas[3]) * (M2Ft **2) # Wing Area (Last 2 Sections) (Ft^2)
Fuselage_wetted_Area = Swet/Sref * (WS_areas[0] + WS_areas[1]) * (M2Ft **2) # Fuselage Area (First 2 Sections) (Ft^2)
Wing_sweep = WS_sweeps[2] # Outer Wing Sweep (deg)
Wing_taper = WS_trs[2] # Outer Wing Taper
Vt_area = 2*VStabalizer_area * (M2Ft **2) # Area of Vertical Tail (Ft^2)
Length_tail = .55 * WS_rootcs[0] * M2Ft # Aproximated (ft)
Length_fuselage = WS_rootcs[0] * M2Ft   # Feet
Diameter_fuselage = (WS_spans[0] + 1.15)*2 * M2Ft # Feet (Includes 1.15m cargo space) 
Payload_lb = Obj['Payload Weight'] / Ntolb # Payload weight (lb)

# Find Weights
Weight_Distributions = Weights.estimate_aircraft_weights(plot_pie=False, export_csv=False, Sw=Wing_wetted_Area, AR=AR,lambda_outer_deg=Wing_sweep,taper=Wing_taper,V_mach=Obj['Mach'], Svt=Vt_area, Sf = Fuselage_wetted_Area, Lt_ft=Length_tail,L_ft=Length_fuselage,D_ft=Diameter_fuselage,payload_lb=Payload_lb,fuel_fraction=Obj['Fuel Fraction'])

# ────────────────────────────────────────────────
# Section 7 - Compute Aero Values
# ────────────────────────────────────────────────
from AeroFunctions import Aero_Driver, compute_density
# Compute Inputs
rho = compute_density.compute(Obj['Altitude'])
MTOW = Weight_Distributions['weights_N']['total']
plotting_flag = False
printing_flag = False

# Get Aero Vals
LD_cruise, CL_cruise, Alpha_cruise, CDi_cruise, CDw_cruise, CD_total_cruise = Aero_Driver.bwb_cruise_analysis(plotting_flag,printing_flag,AR,b,Diameter_fuselage,WS_sweeps,WS_rootcs,Swet/Sref,WS_areas,Obj['Mach'],CD0,rho,MTOW)

# ────────────────────────────────────────────────
# Section 8a - Compute Aircraft CG
# ────────────────────────────────────────────────
from WeightFunctions import CGNPSM

# Compute Inputs
Offsets = np.zeros(len(WS_spans))
Offsets[0] = 0.0  # root section starts at 0 (nose or front reference)
for i in range(1, len(WS_spans)):
    # Cumulative X offset due to previous sweep and span
    prev_span = WS_spans[i-1]
    prev_sweep = WS_sweeps[i-1]
    Offsets[i] = Offsets[i-1] + prev_span * np.tan(np.radians(prev_sweep))

# Compute Center of Gravity
Xcg = CGNPSM.compute_cg(Weight_Distributions['percentages'],WS_trs,WS_rootcs,WS_spans,WS_sweeps,WS_areas*2,Offsets,MAC,Obj['Mach'],LD_cruise,Obj['Range'],Obj['TSFC'])

# ────────────────────────────────────────────────
# Section 8b - Compute AVL Neutral Point
# ────────────────────────────────────────────────
from AVLFunctions import AVL
# Generate AVL File and Compute Xnp
AVL.generate_avl_file(WS_spans, WS_rootcs, WS_tipcs, WS_sweeps, WS_dihedrals,f'{basefilename}.avl',basefilename,Obj['Mach'],Sref,MAC,b,Xcg[0])
Xnp = AVL.get_neutral_point_from_avl(base_name = basefilename, alpha_cruise = Alpha_cruise,avl_executable = r"C:\Users\ajbur\OneDrive\Desktop\School\MAE FILES\BURNS\MAE 155A\BWB_Python\AVLFunctions\avl352.exe",timeout_seconds = 15)

# ────────────────────────────────────────────────
# Section 8c - Compute Static Stability (SM)
# ────────────────────────────────────────────────
SM = 100*(Xnp - Xcg)/MAC

# ────────────────────────────────────────────────
# Section 8d - Compute Dynamic Stability (Derivatives)
# ────────────────────────────────────────────────

# ────────────────────────────────────────────────
# Section 9 - Compute Aircraft Cost
# ────────────────────────────────────────────────
from CostFunctions import Cost
plotting_flag = False
Cost_per_hr = Cost.compute_per_hour_cost(plotting_flag,Obj['TSFC'],LD_cruise,MTOW) # Compute Per Hour Cost

# ────────────────────────────────────────────────
# Section 10 - Save Iteration Data
# ────────────────────────────────────────────────
# 1. Aggregate all data into a single flat dictionary
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
iteration_data = {
    "Timestamp": timestamp,
    "VSP_File": vspfilename,
    
    # High Level Objectives
    "Range_m": Obj['Range'],
    "Mach": Obj['Mach'],
    "Altitude_ft": Obj['Altitude'],
    "Payload_N": Obj['Payload Weight'],
    "Fuel_Fraction": Obj['Fuel Fraction'],
    
    # Shape Parameters (Converted from numpy arrays to lists for easy saving)
    "WS_spans": WS_spans.tolist(),
    "WS_rootcs": WS_rootcs.tolist(),
    "WS_tipcs": WS_tipcs.tolist(),
    "WS_sweeps": WS_sweeps.tolist(),
    "WS_dihedrals": WS_dihedrals.tolist(),
    "WS_trs": WS_trs.tolist(),
    
    # Aero & VSP Values
    "Sref": Sref,
    "Swet": Swet,
    "MAC": MAC,
    "AR": AR,
    "CD0": CD0,
    
    # Performance & Cruise
    "MTOW_N": MTOW,
    "LD_cruise": LD_cruise,
    "CL_cruise": CL_cruise,
    "Alpha_cruise": Alpha_cruise,
    "CDi_cruise": CDi_cruise,
    "CDw_cruise": CDw_cruise,
    "CD_total_cruise": CD_total_cruise,
    
    # Stability
    "CG": Xcg,
    "NP": Xnp,
    "Static_Margin": SM,

    # Cost
    "Cost_Per_Hour": Cost_per_hr
}

# ---------------------------------------------------------
# Adds a new row to a master file for plotting/trade studies
# ---------------------------------------------------------
csv_filename = "BWB_Iteration_Log.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=iteration_data.keys())
    
    # Write the header only if the file is brand new
    if not file_exists:
        writer.writeheader()
        
    writer.writerow(iteration_data)
print(f"Appended iteration data to {csv_filename}")