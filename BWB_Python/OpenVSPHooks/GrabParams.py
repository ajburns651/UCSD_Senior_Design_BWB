# ────────────────────────────────────────────────
# 155A BWB - OpenVSP - Grab Parameters
# ────────────────────────────────────────────────
import openvsp as vsp
import numpy as np
import os
import sys

### Grab Plane Sizing Dimensions ###
def sizing(VSPFILE, wing_id, vstabalizer_id, N_WingSections):

    # Suppress VSP/stdout output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull

    vsp.ClearVSPModel()  # Removes ALL current geoms, resets to blank state
    vsp.Update()
    vsp.ReadVSPFile(VSPFILE) # Load In Updated VSP File
    vsp.Update()

    ## Full Model Values
    # Grab MAC & AR
    MAC = vsp.GetParmVal(vsp.FindParm(wing_id, "MAC", "WingGeom"))
    AR = vsp.GetParmVal(vsp.FindParm(wing_id, "TotalAR", "WingGeom"))
    b = vsp.GetParmVal(vsp.FindParm(wing_id, "TotalProjectedSpan", "WingGeom"))

    # Comp_Geom (Wetted Area)
    vsp.ComputeCompGeom(vsp.SET_ALL, False, 0)
    vsp.Update()
    comp_res_id = vsp.FindLatestResultsID("Comp_Geom")

    Swet = vsp.GetDoubleResults(comp_res_id, "Wet_Area", 0)[0]

    ## Find Values Per Section
    # Pre Allocate Arrays Of Values For Wing Sections
    WS_Areas = np.zeros(N_WingSections)

    for i in range(N_WingSections):
        WS_Areas[i] = vsp.GetParmVal(vsp.FindParm(wing_id, "Area", f"XSec_{i+1}")) 

    Sref = sum(WS_Areas) * 2

    ## Get Vertical Stabalizer Parameters
    VStabalizer_area = vsp.GetParmVal(vsp.FindParm(vstabalizer_id, "Area", "XSec_1")) 

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    devnull.close()

    return Sref, Swet, WS_Areas, VStabalizer_area, MAC, AR, b

### Perform Parasite Drag Calculation ###
def parasite(VSPFILE, Altitude, Mach):
    # Load File
    vsp.ClearVSPModel()  # Removes ALL current geoms, resets to blank state
    vsp.Update()
    vsp.ReadVSPFile(VSPFILE) # Load In Updated VSP File
    vsp.Update()

    # Define Drag Simulation Parameters
    vsp.SetDoubleAnalysisInput("ParasiteDrag", "Altitude", [Altitude], 0)          
    vsp.SetDoubleAnalysisInput("ParasiteDrag", "Mach", [Mach], 0)

    # Run the CD0 Analysis
    rid = vsp.ExecAnalysis("ParasiteDrag")

    # Get Results
    CD0 = vsp.GetDoubleResults(rid, "Total_CD_Total", 0)[0]

    return CD0