import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# 1. Load the Data
# Replace with your actual filename if it's different
csv_file = "Monte_Carlo_Results_Expanded.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Could not find {csv_file}. Make sure you ran the Monte Carlo script first!")
    exit()

# Filter out any completely failed/NaN runs just in case
df = df.dropna()

import re

# Function to extract the first number from an array string
def extract_first_number(val):
    if isinstance(val, str):
        # Finds all numbers (including negatives and decimals) in the string
        numbers = re.findall(r'-?\d+\.\d+|-?\d+', val)
        if numbers:
            return float(numbers[0]) # Change [0] to a different index if cruise isn't the first value
    return float(val)

# Clean up the Static Margin column
df['Static_Margin'] = df['Static_Margin'].apply(extract_first_number)

# Set the visual style
sns.set_theme(style="whitegrid")

# ==========================================
# PLOT 1: Pareto Frontier (L/D vs Cost + SM)
# ==========================================
plt.figure(figsize=(10, 6))
# We plot L/D on X and MTOW on Y. We color the points by Static Margin.
scatter = plt.scatter(df['L_D_Cruise'], df['MTOW'],
                      c=df['Static_Margin']*100, cmap='viridis', vmin=-40, vmax=40,
                      alpha=0.7, edgecolors='w', linewidth=0.5)

plt.colorbar(scatter, label='Static Margin')
plt.title('N = 1000 Random Simulations', fontsize=14, fontweight='bold')
plt.ylabel('MTOW (N)', fontsize=12)
plt.xlabel('Cruise Efficiency (L/D)', fontsize=12)
plt.tight_layout()
plt.savefig("Plot_1_Pareto.png", dpi=300)
plt.show()

# ==========================================
# PLOT 2: Correlation Heatmap
# ==========================================
plt.figure(figsize=(12, 10))
# Calculate the correlation matrix
columns_to_exclude = ['Run_ID', 'Cargo Chord', 'Wing Chord']

rename_dict = {
    'Static_Margin': 'SM',
    'L_D_Cruise': 'L/D',           # Note: original had underscore
    'Span_Total': 'Span',
    'Fuselage Chord': 'Chord',
    'Sweep_inner': 'Inner Sweep',
    'Sweep_outer': 'Outer Sweep',
    'Cost_Per_Hour': 'Cost'
    # MTOW is left unchanged — add here if you want to rename it too (e.g. 'MTOW': 'MTOW')
}

# ────────────────────────────────────────────────
# Step 2: Create filtered & renamed DataFrame
# ────────────────────────────────────────────────
# Drop unwanted columns
df_filtered = df.drop(columns=columns_to_exclude, errors='ignore')

# Rename the desired columns (only those that exist)
df_filtered = df_filtered.rename(columns=rename_dict)

# Calculate correlation matrix only on the remaining numeric columns
corr = df_filtered.corr()

order = ['Cost', 'L/D', 'Span', 'MTOW', 'Chord', 'SM', 'Inner Sweep', 'Outer Sweep']
corr = corr.loc[order, order]

# Create a heatmap
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
            cbar=True, square=True, linewidths=0.5,
            vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("Plot_2_Correlation.png", dpi=300)
plt.show()

# ==========================================
# PLOT 3: Output Distributions (Histograms)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MTOW Histogram
sns.histplot(df['MTOW'], bins=30, kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribution of MTOW', fontsize=12, fontweight='bold')
axes[0].set_xlabel('MTOW (Newtons)')
axes[0].set_ylabel('Frequency')

# Static Margin Histogram
sns.histplot(df['Static_Margin'], bins=30, kde=True, ax=axes[1], color='orange')
axes[1].set_title('Distribution of Static Margin', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Static Margin')
axes[1].set_ylabel('Frequency')

# Add a red line at 0 for Static Margin to show unstable designs
axes[1].axvline(0, color='red', linestyle='--', label='Neutral Stability (0)')
axes[1].legend()

plt.tight_layout()
plt.savefig("Plot_3_Distributions.png", dpi=300)
plt.show()

# ==========================================
# PLOT 4: Sensitivity (L/D vs Wingspan)
# ==========================================
plt.figure(figsize=(8, 5))
# A regression plot shows the scatter dots PLUS a line of best fit
# First: scatter with hue (no regression)
sns.scatterplot(
    data=df,
    x='Span_Total',
    y='L_D_Cruise',
    hue='Static_Margin',
    hue_norm=(-0.4, 0.4),               # this clips colors
    palette='viridis',
    alpha=0.7,
    edgecolor='w',
    linewidth=0.5,
    s=60,
    legend=False                        # ← turn off automatic legend
)

# Create matching normalization & colorbar manually
norm = mcolors.Normalize(vmin=-40, vmax=40, clip=True)
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])   # required trick for colorbar without scatter reference

cbar = plt.colorbar(sm, ax=plt.gca(), aspect=30, pad=0.03)
cbar.set_label('Static Margin', fontsize=11)

# Second: overlay regression (no scatter, just the line + CI)
sns.regplot(
    data=df,
    x='Span_Total',
    y='L_D_Cruise',
    scatter=False,            # ← important: turn off its own scatter
    order=2,                  # quadratic
    line_kws={'color': 'red', 'lw': 2},
    ci=95                     # or None
)

plt.title('Sensitivity: Cruise Efficiency vs. Wingspan', fontsize=14, fontweight='bold')
plt.ylabel('Lift-to-Drag Ratio (L/D)', fontsize=12)
plt.xlabel('Wingspan (m)', fontsize=12)
plt.tight_layout()
plt.savefig("Plot_4_Sensitivity.png", dpi=300)
plt.show()

# ==========================================
# PLOT 5: Pareto Frontier (L/D vs MTOW)
# ==========================================
plt.figure(figsize=(10, 6))
# We plot L/D on X and MTOW on Y. We color the points by Static Margin.
scatter = plt.scatter(df['L_D_Cruise'], df['MTOW'], 
                      c=df['Cost_Per_Hour'], cmap='viridis', 
                      alpha=0.7, edgecolors='w', linewidth=0.5)

plt.colorbar(scatter, label='Cost Per Hour ($)')
plt.title('BWB Design Trade-offs: Aerodynamics vs Weight', fontsize=14, fontweight='bold')
plt.xlabel('Lift-to-Drag Ratio (L/D)', fontsize=12)
plt.ylabel('Maximum Takeoff Weight (N)', fontsize=12)
plt.tight_layout()
plt.savefig("Plot_5_Pareto.png", dpi=300)
plt.show()

print("All plots generated and saved as PNG files!")