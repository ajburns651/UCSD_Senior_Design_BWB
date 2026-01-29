import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf
import pandas as pd

## NACA 5 Digit Airfoil (Lift coef, Location of Max Camber, Camber Shape Type, Thickness (% of chord))
def naca_5_digit(designation, n_points=200):
    # Generates coordinates for a NACA 5-digit airfoil. designation: string like "23012"
    cld = int(designation[0]) * 0.15
    p = int(designation[1]) / 20
    reflex = int(designation[2]) == 1
    t = int(designation[3:]) / 100

    15009

    x = np.linspace(0, 1, n_points) # Generate number of coordinaes to generate along chord

    # Thickness distribution For NACA 5 Digit
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    # Pre Allocate Camber Line Vectors
    yc = np.zeros_like(x);dyc_dx = np.zeros_like(x)

    k1 = 6 * cld / (p**3 * (1 - p))
    for i, xi in enumerate(x):
        if xi < p:
            yc[i] = k1 / 6 * (xi**3 - 3*p*xi**2 + p**2*(3 - p)*xi)
            dyc_dx[i] = k1 / 6 * (3*xi**2 - 6*p*xi + p**2*(3 - p))
        else:
            yc[i] = k1 / 6 * p**3 * (1 - xi)
            dyc_dx[i] = -k1 / 6 * p**3

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])

    return np.column_stack([x_coords, y_coords])

# === Define Flow Conditions ===
Re = 1e8
Mach = 0.85
Alpha = 3.0; # Cruise alpha estimated at 3 degrees AoA

# === Define Constraint For Main Cabin Thickness ===
#min_t_over_c = 0.085 # From T/C Calculated on Airfoil Sizing Methodology
min_t_over_c = 0.085
min_t_over_c_range = np.linspace(0.13, 0.6, 40) # From Airfoil Sizing Methodology, 13% of Chord to 60% of Chord is Cabin

# === Airfoils To Test ===
naca_camber_groups = ["110","120","130","140","150", "210","220","230","240","250", "310","320","330","340","350", "410","420","430","440","450", "510","520","530","540","550"]
thickness_vals = range(9, 19)
results = []

# === Main Iteration Loop ===
for camber in naca_camber_groups:
    for t in thickness_vals:
        airfoil_name = f"{camber}{t:02d}"
        coords = naca_5_digit(airfoil_name)

        try:
            airfoil = asb.Airfoil(name=airfoil_name,coordinates=coords) # Grab Airfoil From Aerosandbox
        except:
            continue  # Skip Invalid Airfoils

        # === Verify Thickness Constraint Is Met ===
        thickness = airfoil.local_thickness(min_t_over_c_range)
        if np.min(thickness) < min_t_over_c:
            continue

        # === Compute Aero Parameters From Neuralfoil ===
        aero = airfoil.get_aero_from_neuralfoil(alpha=Alpha,Re=Re,mach=Mach)
        CL = aero["CL"].item(); CD = aero["CD"].item(); CM = aero["CM"].item()

        # === Append Results To Allocated Matrix ===
        results.append({"Airfoil": f"naca{airfoil_name}", "CL": CL, "CD": CD, "CL/CD": CL/CD, "CM": CM, "Min_t_13_60": float(np.min(thickness))})

# === Sort Final Airfoil Results By L/D ===
df = pd.DataFrame(results)
df = df.sort_values("CL/CD", ascending=False)
print(df.head(10))


