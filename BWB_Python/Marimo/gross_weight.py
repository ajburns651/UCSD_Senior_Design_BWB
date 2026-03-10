import numpy as np

def compute_fractions(
    Wp: float,
    R: float,
    aircraft_type: str = "Jet-propelled",
    V: float = 295.1,
    ewf_a: float = 0.97,
    ewf_b: float = -0.06,
    LD: float = 20,
    cT: float = 1.415e-5,
    cP: float = 5e-8,
    ns: float = 0.9,
    nb: float = 0.9,
    eb: float = 7e5,
    ef: float = 4.4e7,
    loiter_min: float = 30,
    cruise_add_min: float = 45,
    Wg0: float = 1e6,
    maxiter: int = 200,
    tol: float = 1e-3,
) -> dict:
    """
    Parameters
    ----------
    Wp            : Payload weight [N]
    R             : Range [m]
    aircraft_type : One of "Jet-propelled", "Turbo-electric",
                    "Turbo-electric (alternate)", "Pure-electric"
    V             : Cruise speed [m/s]
    ewf_a, ewf_b  : Empty weight fraction regression coefficients (We/Wg = a * Wg^b)
    LD            : Lift-to-drag ratio
    cT            : TSFC [kg/N/s]
    cP            : BSFC [kg·s/W]
    ns            : Shaft-to-thrust power efficiency
    nb            : Battery-to-thrust power efficiency
    eb            : Battery specific energy [J/kg]
    ef            : Fuel specific energy [J/kg]
    loiter_min    : Loiter time [min]  (jet-propelled only)
    cruise_add_min: Additional cruise time [min]  (jet-propelled only)
    Wg0           : Initial guess for gross weight [N]
    maxiter       : Maximum solver iterations
    tol           : Convergence tolerance [N]

    Returns
    -------
    dict with keys: Wg, We, Ws, Wp, We_Wg, Ws_Wg, converged, iterations,
                    and (for jet-propelled) wf_mission_segments
    """
    g = 9.81
    loiter  = loiter_min * 60       # convert to seconds
    cruise_add = cruise_add_min * 60

    # --- Energy-storage weight fraction ---
    if aircraft_type == "Jet-propelled":
        wf_takeoff  = 0.970
        wf_climb    = 0.985
        wf_landing  = 0.995
        wf_cruise        = np.exp(-cT * g * R          / (LD * V))
        wf_loiter        = np.exp(-cT * g * loiter     /  LD)
        wf_cruise_add    = np.exp(-cT * g * cruise_add /  LD)
        wf_total = (
            wf_takeoff
            * wf_climb**2
            * wf_cruise
            * wf_loiter**2
            * wf_cruise_add
            * wf_landing**2
        )
        Ws_Wg = 1 - wf_total
        mission_segments = {
            "wf_takeoff":       wf_takeoff,
            "wf_climb":         wf_climb,
            "wf_cruise":        wf_cruise,
            "wf_loiter":        wf_loiter,
            "wf_cruise_add":    wf_cruise_add,
            "wf_landing":       wf_landing,
            "wf_total":         wf_total,
        }
    elif aircraft_type == "Turbo-electric":
        Ws_Wg = 1 - np.exp(-R * cP * g / LD / ns)
        mission_segments = {}
    elif aircraft_type == "Turbo-electric (alternate)":
        Ws_Wg = 1 - np.exp(-R * g / LD / ns / ef)
        mission_segments = {}
    elif aircraft_type == "Pure-electric":
        Ws_Wg = R * g / LD / nb / eb
        mission_segments = {}
    else:
        raise ValueError(f"Unknown aircraft_type: {aircraft_type!r}")

    # --- Fixed-point iteration: Wg = Wp / (1 - We/Wg - Ws/Wg) ---
    Wg = Wg0
    converged = False
    for i in range(maxiter):
        We_Wg = ewf_a * Wg**ewf_b
        rhs = Wp / (1.0 - We_Wg - Ws_Wg)
        residual = Wg - rhs
        Wg -= residual
        if abs(residual) < tol:
            converged = True
            break

    We_Wg = ewf_a * Wg**ewf_b
    return {
        "Wg":                Wg,
        "We":                We_Wg * Wg,
        "Ws":                Ws_Wg * Wg,
        "Wp":                Wp,
        "We_Wg":             We_Wg,
        "Ws_Wg":             Ws_Wg,
        "converged":         converged,
        "iterations":        i + 1,
        "wf_mission_segments": mission_segments,
    }


if __name__ == "__main__":
    result = compute_fractions(Wp=392000, R=1.2964e7, LD = 23.42)
    print(result)