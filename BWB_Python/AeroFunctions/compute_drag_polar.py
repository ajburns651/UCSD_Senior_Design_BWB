## === Script To Compute Base Cl, Clalpha, and Cd w/o wave drag === ##
import numpy as np

# Main Function
def determine_parameters(aoa, geom, aero):
    # Prepare arrays (3 sections: typically centerbody + inboard + outboard)
    CL_sec      = np.zeros((3, len(aoa)))
    CLalpha_sec = np.zeros((3, len(aoa)))
    sweeps = geom['Sweeps']

    for i in range(3):
        # Zero-lift CL (sectional)
        CL0_sec = 0.225 * np.cos(np.deg2rad(sweeps[i]))

        # Centerbody boost (empirical extra camber lift)
        if i == 0:
            CL0_sec += 0.15

        # Prandtl-Glauert compressibility factor
        beta = np.sqrt(1 - aero['M']**2)

        # Sectional efficiency / thickness derate
        eta_sec = (aero['Cla'] * beta) / (2 * np.pi) * (1 - 0.2 * geom['t_c'][i])

        # Approximate local aspect ratio (semi-span basis)
        AR_sec = geom['AR'] * geom['Swr_frac'][i] * 2

        # Helmbold-type lift curve slope (with sweep correction)
        denom = np.sqrt(4 + AR_sec**2 * beta**2 * (1 + np.tan(np.deg2rad(sweeps[i]))**2 / beta**2) / eta_sec**2)
        CLalpha_sec[i, :] = (2 * np.pi * AR_sec) / denom

        # Sectional lift coefficient
        CL_sec[i, :] = CL0_sec + CLalpha_sec[i, :] * aoa

    # Weighted total lift coefficient (by area fraction)
    CL = np.sum(CL_sec * geom['Swr_frac'][:, np.newaxis], axis=0)

    # Weighted total lift curve slope
    CLalpha = np.sum(CLalpha_sec * geom['Swr_frac'][:, np.newaxis], axis=0)

    # Induced drag factor (Oswald efficiency based)
    k = 1.0 / (np.pi * aero['e'] * geom['AR'])

    # Base drag = parasite + induced
    CD_base = aero['Cd0'] + k * (CL ** 2)

    return CL, CLalpha, CD_base