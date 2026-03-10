## === Script To Compute Wave Drag (Full Korn Equation) === ##
import numpy as np

def determine_wave_drag(CL, geom, aero):
    """
    Full Korn equation with sweep applied per section, then area-weighted.

    M_DD_i = ka / cos(Lambda_i)
               - (t/c)_i / cos^2(Lambda_i)
               - CL / (10 * cos^3(Lambda_i))

    ka is constant (supplied in aero['ka']).
    M_DD is area-weighted across sections before applying drag rise.

    Parameters
    ----------
    CL      : array (len(aoa),)  — lift coefficient sweep
    geom    : dict with keys:
                't_c'     — thickness ratio per section, shape (n,)
                'Sweeps'  — sweep angle per section in degrees, shape (n,)
                'Swr_frac'— planform area fraction per section, shape (n,)
    aero    : dict with keys:
                'M'       — cruise Mach number (scalar)
                'ka'      — Korn technology factor (scalar, constant)
    """

    M      = aero['M']
    ka     = aero['ka']
    t_c    = geom['t_c']                                    # shape (n,)
    sweeps = np.deg2rad(geom['Sweeps'])                     # shape (n,)
    w      = geom['Swr_frac'] / np.sum(geom['Swr_frac'])   # normalized weights

    cos1   = np.cos(sweeps)                                 # shape (n,)
    cos2   = cos1**2
    cos3   = cos1**3

    # ------------------------------------------------------------------
    # Korn equation per section: M_DD_i for each CL value
    # M_DD_sec shape: (n_sections, len(CL))
    # ------------------------------------------------------------------
    M_DD_sec = (ka / cos1[:, np.newaxis]
                - t_c[:, np.newaxis] / cos2[:, np.newaxis]
                - CL[np.newaxis, :] / (10.0 * cos3[:, np.newaxis]))

    # Area-weighted M_DD across sections → shape (len(CL),)
    M_DD = np.sum(M_DD_sec * w[:, np.newaxis], axis=0)

    # ------------------------------------------------------------------
    # Drag rise: apply only where M exceeds local divergence Mach
    # CD_wave = 20 * (M - M_DD)^4   [Korn/Lock drag rise model]
    # ------------------------------------------------------------------
    CD_wave         = np.zeros_like(CL)
    mask            = M > M_DD
    CD_wave[mask]   = 20.0 * (M - M_DD[mask])**4

    return CD_wave