# Compute Density Given Altitude in Feet
import numpy as np

def compute(Altitude):
    altitude_m = np.asarray([Altitude * .3048])
    
    # Constants
    g = 9.81
    R = 287
    T0 = 288.15          # Sea-level temp (K)
    L_trop = 0.0065      # Troposphere lapse (K/m)
    h_trop = 11000.0     # Tropopause (m)
    T_trop = T0 - L_trop * h_trop  # 216.65 K
    
    h_strat1 = 20000.0
    h_strat2 = 32000.0
    h_strat3 = 47000.0
    
    # Temperature array
    T = np.full_like(altitude_m, T_trop, dtype=float)
    
    # Troposphere (0–11 km)
    mask_trop = altitude_m <= h_trop
    T[mask_trop] = T0 - L_trop * altitude_m[mask_trop]
    
    # Tropopause to 20 km: isothermal at T_trop (already set)
    # No change needed
    
    # 20–32 km: +1 K/km
    mask_strat1 = (altitude_m > h_strat1) & (altitude_m <= h_strat2)
    if np.any(mask_strat1):
        T[mask_strat1] = T_trop + 0.001 * (altitude_m[mask_strat1] - h_strat1)
    
    # 32–47 km: +2.8 K/km
    mask_strat2 = (altitude_m > h_strat2) & (altitude_m <= h_strat3)
    if np.any(mask_strat2):
        # Temperature at start of this layer (32 km)
        T_at_32km = T_trop + 0.001 * (h_strat2 - h_strat1)
        T[mask_strat2] = T_at_32km + 0.0028 * (altitude_m[mask_strat2] - h_strat2)
    
    # Above 47 km: +2.0 K/km (simplified – extend if needed)
    mask_upper = altitude_m > h_strat3
    if np.any(mask_upper):
        T_at_47km = T_at_32km + 0.0028 * (h_strat3 - h_strat2)
        T[mask_upper] = T_at_47km + 0.0020 * (altitude_m[mask_upper] - h_strat3)
    
    # Full density calculation (pressure ratio delta + ideal gas law)
    # Simplified troposphere-only version – expand for full accuracy if needed
    delta = np.ones_like(T)
    mask_trop = altitude_m <= h_trop
    delta[mask_trop] = (T[mask_trop] / T0) ** (g / (R * L_trop))
    
    # Isothermal layers (tropopause & stratosphere)
    # Use exponential decay for pressure
    mask_iso_layers = altitude_m > h_trop
    if np.any(mask_iso_layers):
        delta_trop = (T_trop / T0) ** (g / (R * L_trop))
        delta[mask_iso_layers] = delta_trop * \
                                 np.exp(-g * (altitude_m[mask_iso_layers] - h_trop) / (R * T_trop))
    
    rho = 1.225 * delta * (T0 / T)  # rho = rho0 * delta * (T0 / T)
    
    return rho[0] if len(altitude_m) == 1 else rho