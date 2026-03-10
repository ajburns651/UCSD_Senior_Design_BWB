## === Script To Compute Wave Drag === ##
import numpy as np


def determine_wave_drag(CL, geom, aero):
    # ----- Parameters -----
    t_c_avg = np.mean(geom['t_c'])           # average thickness ratio
    M       = aero['M']                      # cruise Mach number
    k       = 1500.0                         # empirical constant
    ka      = 0.95                           # conventional airfoil factor
    n       = 3.0                            # exponent for transonic drag rise

    # ----- Critical Mach (divergence Mach) -----
    M_DD = ka - 0.1 * CL - t_c_avg

    # ----- Wave drag calculation -----
    CD_wave = np.zeros_like(CL)

    # Only apply wave drag when cruise Mach > local divergence Mach
    mask = M > M_DD
    CD_wave[mask] = k * t_c_avg**2 * (M - M_DD[mask])**n

    return CD_wave