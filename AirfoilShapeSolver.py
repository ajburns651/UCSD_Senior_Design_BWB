import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

Re = 1e8
mach = 0.85

initial_guess_airfoil = asb.KulfanAirfoil("naca4412")
initial_guess_airfoil.name = "Initial Guess (NACA4412)"

opti = asb.Opti()

# === Set Design Variables ===
optimized_airfoil = asb.KulfanAirfoil(
    name="Optimized",
    lower_weights=opti.variable(
        init_guess=initial_guess_airfoil.lower_weights,
        lower_bound=-0.5,
        upper_bound=0.25,
    ),
    upper_weights=opti.variable(
        init_guess=initial_guess_airfoil.upper_weights,
        lower_bound=-0.25,
        upper_bound=0.5,
    ),
    leading_edge_weight=opti.variable(
        init_guess=initial_guess_airfoil.leading_edge_weight,
        lower_bound=-0.5,
        upper_bound=0.5,
    ),
    TE_thickness=0,
)

alpha = opti.variable(init_guess=0, lower_bound=-20, upper_bound=20)

aero = optimized_airfoil.get_aero_from_neuralfoil(
    alpha=alpha,
    Re=Re,
    mach=mach,
)

opti.subject_to(
    [
        aero["analysis_confidence"] > 0.95,
        optimized_airfoil.lower_weights[0] < -0.05,
        optimized_airfoil.upper_weights[0] > 0.05,
        optimized_airfoil.local_thickness(np.linspace(0.01, 0.99)) > 0,
        optimized_airfoil.TE_angle() >= 0,
        np.abs(aero["CM"]) <= 0.05
    ]
)

get_wiggliness = lambda af: sum(
    [
        np.sum(np.diff(np.diff(array)) ** 2)
        for array in [af.lower_weights, af.upper_weights]
    ]
)

opti.subject_to(
    get_wiggliness(optimized_airfoil) < 2 * get_wiggliness(initial_guess_airfoil)
)

# Objective: maximize L/D with optional soft penalty on CM (optional)
penalty = 1e-2  # small weight for CM penalty
opti.maximize(aero["CL"] / aero["CD"] - penalty * np.abs(aero["CM"]))

airfoil_history = []
aero_history = []

def callback(i):
    airfoil_history.append(
        asb.KulfanAirfoil(
            name="in-progress",
            lower_weights=opti.debug.value(optimized_airfoil.lower_weights),
            upper_weights=opti.debug.value(optimized_airfoil.upper_weights),
            leading_edge_weight=opti.debug.value(optimized_airfoil.leading_edge_weight),
            TE_thickness=opti.debug.value(optimized_airfoil.TE_thickness),
        )
    )
    aero_history.append({k: opti.debug.value(v) for k, v in aero.items()})

sol = opti.solve(
    callback=callback,
    behavior_on_failure="return_last",
    options={"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"},
)

optimized_airfoil = sol(optimized_airfoil)
aero = sol(aero)

# === Print aerodynamic results ===
print("=== Optimized Airfoil Results ===")
print(f"CL = {sol(aero['CL']):.4f}")
print(f"CD = {sol(aero['CD']):.6f}")
print(f"CM = {sol(aero['CM']):.4f}")
print(f"CL/CD = {sol(aero['CL']/aero['CD']):.2f}")
print(f"Top Xtr = {sol(aero.get('Top_Xtr', 'N/A'))}")
print(f"Bot Xtr = {sol(aero.get('Bot_Xtr', 'N/A'))}")
print(f"Analysis Confidence = {sol(aero['analysis_confidence']):.2f}")



# === Graph Airfoil Differences ===

fig, ax = plt.subplots(figsize=(6, 2))

# Get coordinates
opt_coords = optimized_airfoil.coordinates
init_coords = initial_guess_airfoil.coordinates

# Plot
ax.plot(opt_coords[:, 0], opt_coords[:, 1], label="Optimized Airfoil", color="blue")
ax.plot(init_coords[:, 0], init_coords[:, 1], label="Initial Guess", color="red")

# Labels, legend, title
ax.set_xlabel("x/c")
ax.set_ylabel("y/c")
ax.legend()
ax.set_title("Airfoil Optimization")

plt.show()

