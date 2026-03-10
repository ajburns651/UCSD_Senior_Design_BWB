## === Script To Compute Base CL, CLalpha, and CD w/o wave drag === ##
import numpy as np

def determine_parameters(aoa, geom, aero):
    """
    Inputs (per section, shape (n_sections,)):
        aero['CLalpha_sec']  — 2D lift slope per rad
        aero['CL0_sec']      — 2D zero-alpha lift coefficient
        aero['Cd0_sec']      — 2D parasite drag per section
        geom['Swet_Sref']    — wetted area / Sref per section (array)
        geom['Swr_frac']     — planform area fraction per section
        geom['Sweeps']       — sweep angle per section (deg)
        geom['AR']           — full wing aspect ratio
        aero['e']            — Oswald efficiency
        aero['M']            — cruise Mach
    """

    beta = np.sqrt(1 - aero['M']**2)                           # Prandtl-Glauert factor
    w    = geom['Swr_frac'] / np.sum(geom['Swr_frac'])         # normalized area weights

    # ------------------------------------------------------------------
    # 1. Lift curve slope: 2D → compressible + swept → 3D (Helmbold)
    # ------------------------------------------------------------------
    # PG compressibility + sweep correction on each section's 2D slope
    CLalpha_sec_swept = (aero['Clalpha_sec'] / beta) * np.cos(np.deg2rad(geom['Sweeps']))

    # Area-weighted effective 2D slope (input to Helmbold)
    CLalpha_2D_eff = np.sum(CLalpha_sec_swept * w)

    # Helmbold relation → true 3D finite-wing lift slope
    kH         = CLalpha_2D_eff / (np.pi * geom['AR'])
    CLalpha_3D = CLalpha_2D_eff / (np.sqrt(1.0 + kH**2) + kH)

    # ------------------------------------------------------------------
    # 2. Zero-lift CL: scale 2D CL0 by (3D/2D slope ratio)
    #    Preserves zero-lift angle while reflecting finite-AR reduction
    # ------------------------------------------------------------------
    CL0_2D_eff = np.sum(aero['Cl0_sec'] * w)
    CL0_3D     = CL0_2D_eff * (CLalpha_3D / CLalpha_2D_eff)

    # ------------------------------------------------------------------
    # 3. 3D lift polar over aoa sweep
    # ------------------------------------------------------------------
    CL      = CL0_3D + CLalpha_3D * aoa        # shape (len(aoa),)
    CLalpha = CLalpha_3D                        # scalar

    # ------------------------------------------------------------------
    # 4. Parasite drag: per-section cd0 scaled by wetted area fraction
    #    CD0 = Σ cd0_sec_i * (Swet_i / Sref)
    #    geom['Swet_Sref'] is a per-section array
    # ------------------------------------------------------------------
    CD0 = np.sum(aero['Cd0_sec'] * w)

    # ------------------------------------------------------------------
    # 5. Base drag polar (no wave drag)
    # ------------------------------------------------------------------
    k_ind   = 1.0 / (np.pi * aero['e'] * geom['AR'])
    CD_base = CD0 + k_ind * CL**2

    return CL, CLalpha, CD_base