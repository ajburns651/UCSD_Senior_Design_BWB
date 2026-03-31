## === Script To Compute CG and SM === ##
import numpy as np

# Main Function
def compute_cg(weights_dict, Taper_Ratios, Root_Chords, Wing_Spans, Sweep_Angles, S, Offset, mac, Mach, LD, Range, TSFC):
    # ────────────────────────────────────────────────
    # Solve For Quarter Chord Of Each Wing
    # ────────────────────────────────────────────────
    def front_plane_coord(percent=0.25):
        # Mean Aerodynamic Chord length per section
        MACs = Root_Chords * (2/3) * (1 + Taper_Ratios + Taper_Ratios**2) / (1 + Taper_Ratios)
        MACs_at_percent = percent * MACs

        # Spanwise location of MAC
        y_mac = (Wing_Spans / 6) * (1 + 2 * Taper_Ratios) / (1 + Taper_Ratios)

        # Sweep offset to MAC location
        sweep_offset = y_mac * np.tan(np.deg2rad(Sweep_Angles))

        # Location relative to section root + aircraft front offset
        x_c = MACs_at_percent + sweep_offset + Offset
        return x_c

    A1_dcldalpha =  4
    A2_dcldalpha =  6
  
    Xc_quarter = front_plane_coord(percent=0.25)

    del_Xac = 0.26*(Mach - .4) ** (2.5)

    Xac = Xc_quarter + del_Xac*np.sqrt(S)

    numerator = A1_dcldalpha*S[0]*Xac[0] + A1_dcldalpha*S[1]*Xac[1] + A2_dcldalpha*S[2]*Xac[2]
    denomenator = A1_dcldalpha*S[0] + A1_dcldalpha*S[1] + A2_dcldalpha*S[2]
    Xnp = numerator/denomenator

    # ────────────────────────────────────────────────
    # Extract relevant weights (in Newtons)
    # ────────────────────────────────────────────────
    W = weights_dict

    W_wing_fuselage     = W.get('wing', 0) + W.get('fuselage', 0)
    W_both_engines      = W.get('installed_engine', 0)
    W_flight_controls   = W.get('flight_controls', 0)
    W_landing_gear      = W.get('main_landing_gear', 0)
    W_air_conditioning  = W.get('air_conditioning_anti_ice', 0)
    W_furnishings       = W.get('furnishings', 0)
    W_fuel              = W.get('fuel', 0)
    W_payload           = W.get('payload', 0)

    # ────────────────────────────────────────────────
    # Compute Fuel Fractions Across Mission (in Newtons)
    # ────────────────────────────────────────────────
    V = Mach * 295.07; R = Range; g = 9.81; cT = TSFC
    loiter = 30 * 60
    cruise_add = 45 * 60


    wf_takeoff = 0.970
    wf_climb = 0.985
    wf_landing = 0.995
    wf_cruise = np.exp(-cT * g * R / (LD * V))
    wf_loiter = np.exp(-cT * g * loiter / LD)
    wf_additional_cruise = np.exp(-cT * g * cruise_add / LD)

    fuel_fractions = np.array([wf_takeoff, wf_climb, wf_cruise, wf_loiter, wf_landing, wf_climb, wf_additional_cruise, wf_loiter, wf_landing])
    remaining_fuel_fraction = np.concatenate(([1.0], np.cumprod(fuel_fractions)+ (.36-1)/.36))
    fuel_weight_at_each_phase = remaining_fuel_fraction * W_fuel

    # ────────────────────────────────────────────────
    # Component CG assumptions
    # ────────────────────────────────────────────────
    Xcg = np.zeros(len(fuel_weight_at_each_phase))
    for i in range(len(fuel_weight_at_each_phase)):
        # Wing parts distributed by area
        wing_weights = W_wing_fuselage * (S / S.sum())
        wing_cg_locs = front_plane_coord(0.40)          # 40% chord

        engine_weight = W_both_engines
        engine_cg_loc = 36.61                           # m

        flight_mid_weight = W_flight_controls * 0.75
        flight_mid_cg_loc = 30.0
        flight_tail_weight = W_flight_controls * 0.25
        flight_tail_cg_loc = 35.8

        gear_weight = W_landing_gear
        gear_cg_loc = Root_Chords[0] / 2                         # middle of aircraft

        cabin_weight = W_air_conditioning + W_furnishings + W_payload
        cabin_cg_loc = 16.475                           # middle of passenger cabin

        fuel_weight = fuel_weight_at_each_phase[i]
        fuel_cg_loc = 24.0

        # ────────────────────────────────────────────────
        # Combine all contributing components
        # ────────────────────────────────────────────────
        C_weights = np.concatenate([
            wing_weights,                       # 4 sections
            [engine_weight],
            [flight_mid_weight],
            [flight_tail_weight],
            [gear_weight],
            [cabin_weight],
            [fuel_weight]
        ])

        C_locations = np.concatenate([
            wing_cg_locs,                       # 4 values
            [engine_cg_loc],
            [flight_mid_cg_loc],
            [flight_tail_cg_loc],
            [gear_cg_loc],
            [cabin_cg_loc],
            [fuel_cg_loc]
        ])
        
        # Aircraft CG location (m)
        total_moment = np.sum(C_weights * C_locations)
        total_weight = np.sum(C_weights)
        Xcg[i] = total_moment / total_weight

    return Xcg