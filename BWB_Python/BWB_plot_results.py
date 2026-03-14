import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOAD & CLEAN
# ─────────────────────────────────────────────
csv_file = "Monte_Carlo_Results_Expanded.csv"
df = pd.read_csv(csv_file).dropna()

def extract_first_number(val):
    if isinstance(val, str):
        nums = re.findall(r'-?\d+\.?\d*', val)
        return float(nums[0]) if nums else np.nan
    return float(val)

for col in ['Static_Margin', 'Spans', 'Sweeps', 'Root_Chords', 'Tip_Chords', 'Taper Ratios']:
    if col in df.columns:
        df[col] = df[col].apply(extract_first_number)

df['MTOW_kg']    = df['MTOW'] / 9.81
df['Stress_Pct'] = df['Stress_Ratio'] * 100

stress_cmap = mcolors.LinearSegmentedColormap.from_list(
    'stress',
    [(0.0, '#2ecc71'), (0.8, '#f1c40f'), (1.0, '#e74c3c')],
    N=512
)
stress_norm = mcolors.Normalize(vmin=0, vmax=1, clip=True)

STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f5f6fa',
    'axes.edgecolor':   '#cccccc',
    'axes.labelcolor':  '#222222',
    'xtick.color':      '#444444',
    'ytick.color':      '#444444',
    'text.color':       '#222222',
    'grid.color':       '#dddddd',
    'grid.linestyle':   '--',
    'grid.alpha':       0.7,
    'font.family':      'monospace',
}
plt.rcParams.update(STYLE)

# ── Globally larger text ──
TITLE_KW = dict(fontsize=16, fontweight='bold', color='#111111', pad=14)
LABEL_KW = dict(fontsize=14)
TICK_SZ  = 13
LEGEND_SZ = 12
ANNOT_SZ  = 11
CBAR_SZ   = 12


# ─────────────────────────────────────────────
# HELPER: draw colorbar
# ─────────────────────────────────────────────
def stress_colorbar(ax, label='Stress Ratio'):
    sm = cm.ScalarMappable(cmap=stress_cmap, norm=stress_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label(label, fontsize=CBAR_SZ)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.yaxis.set_tick_params(color='#444444')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#444444', fontsize=TICK_SZ - 1)
    cbar.ax.axhline(1.0, color='#333333', lw=1.5, ls='--')
    return cbar


# ══════════════════════════════════════════════════════════
# PLOT 1 — L/D vs Wingspan, colored by Stress Ratio
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')

sc = ax.scatter(
    df['Wingspan'], df['L_D_Cruise'],
    c=df['Stress_Ratio'], cmap=stress_cmap, norm=stress_norm,
    s=35, alpha=0.75, edgecolors='none'
)

z = np.polyfit(df['Wingspan'], df['L_D_Cruise'], 2)
x_fit = np.linspace(df['Wingspan'].min(), df['Wingspan'].max(), 300)
ax.plot(x_fit, np.polyval(z, x_fit), color='#333333', lw=2.0, ls='--',
        alpha=0.6, label='Quadratic fit')

failed = df[df['Stress_Ratio'] > 1.0]
ax.scatter(failed['Wingspan'], failed['L_D_Cruise'],
           s=60, facecolors='none', edgecolors='#e74c3c', lw=1.0,
           alpha=0.5, label=f'Overstressed (n={len(failed)})')

stress_colorbar(ax)
ax.axvline(64.0, color='#2980b9', lw=1.4, ls=':', alpha=0.7,
           label='Code E Gate Restriction (65 m)')
ax.set_title('Cruise Efficiency vs. Wingspan', **TITLE_KW)
ax.set_xlabel('Wingspan (m)', **LABEL_KW)
ax.set_ylabel('Lift-to-Drag Ratio (L/D)', **LABEL_KW)
ax.tick_params(axis='both', labelsize=TICK_SZ)
ax.legend(fontsize=LEGEND_SZ, framealpha=0.6)
plt.tight_layout()
plt.savefig('Plot1_LD_vs_Wingspan.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✓ Plot 1 saved")


# ══════════════════════════════════════════════════════════
# PLOT 2 — Correlation Heatmap (lower triangle, no L/D row,
#           no Cruise AoA)
# ══════════════════════════════════════════════════════════
rename = {
    'L_D_Cruise':   'L/D',
    'Wingspan':     'Span',
    'MTOW':         'MTOW',
    'Static_Margin':'Stab. Margin',
    'Stress_Ratio': 'Stress Ratio',
    'Cost_Per_Hour':'Cost/seat-mile',
    'AR':           'AR',
    'Root_Chords':  'Chord',
    'Swet':         'Swet',
    # Alpha_Cruise intentionally excluded
}
scalar_cols = [c for c in rename if c in df.columns]
df_corr = df[scalar_cols].rename(columns=rename)
corr_full = df_corr.corr()

all_vars = ['L/D', 'Span', 'MTOW', 'Stab. Margin', 'Stress Ratio',
            'Cost/seat-mile', 'AR', 'Chord', 'Swet']
row_vars = [v for v in all_vars if v != 'L/D']
col_vars = all_vars

corr_rect = corr_full.loc[row_vars, col_vars]

var_order = {v: i for i, v in enumerate(all_vars)}
mask = np.array([
    [var_order[col] >= var_order[row] for col in col_vars]
    for row in row_vars
], dtype=bool)

fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor('white')

coolwarm_dark = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(
    corr_rect, ax=ax,
    mask=mask,
    annot=True, fmt='.2f', annot_kws={'size': ANNOT_SZ + 1, 'color': '#111111'},
    cmap=coolwarm_dark, center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor='white',
    cbar_kws={'shrink': 0.7, 'pad': 0.02}
)
ax.set_title('Parameter Correlation Matrix (Lower Triangle)', **TITLE_KW)
ax.tick_params(axis='x', rotation=30, labelsize=TICK_SZ)
ax.tick_params(axis='y', rotation=0,  labelsize=TICK_SZ)
ax.figure.axes[-1].tick_params(labelsize=TICK_SZ - 1)   # colorbar ticks
plt.tight_layout()
plt.savefig('Plot2_Correlation.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✓ Plot 2 saved")


# ══════════════════════════════════════════════════════════
# PLOT 3 — Tornado Chart: Spearman Sensitivity Analysis
# ══════════════════════════════════════════════════════════
from scipy.stats import spearmanr
from matplotlib.patches import Patch

outputs = {
    'L/D Cruise':    'L_D_Cruise',
    'Cost/seat-mile':'Cost_Per_Hour',
}

exclude = set(outputs.values()) | {'MTOW_kg', 'Stress_Pct', 'MTOW', 'Static_Margin', 'Root_Moment_Sec3', 'Tip_Chords'}
input_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]

input_labels = {
    'Wingspan':           'Wingspan',
    'AR':                 'Aspect Ratio',
    'Sweeps':             'Inner Sweep',
    'Root_Chords':        'Chord',
    'Stress_Ratio':       'Stress Ratio',
    'Spans':              'Span Segments',
    'Taper Ratios':       'Taper Ratio',
    'Alpha_Cruise':       'Cruise AoA',
    'CL_Cruise':          'CL Cruise',
    'CD0':                'CD0',
    'Fuel_Fraction':      'Fuel Fraction',
    'Swet':               'Swet',
    'Section3_Stress_Pa': 'Bending Stress',
}

records = []
for out_label, out_col in outputs.items():
    if out_col not in df.columns:
        continue
    for in_col in input_cols:
        r, p = spearmanr(df[in_col], df[out_col])
        records.append({'Output': out_label,
                        'Input': input_labels.get(in_col, in_col),
                        'r': r, 'p': p})

sens_df = pd.DataFrame(records)

TOP_N = 10
fig, axes = plt.subplots(1, len(outputs), figsize=(24, 10), sharey=False)
fig.patch.set_facecolor('white')

for ax, (out_label, _) in zip(axes, outputs.items()):
    sub = (sens_df[sens_df['Output'] == out_label]
           .assign(abs_r=lambda d: d['r'].abs())
           .nlargest(TOP_N, 'abs_r')
           .sort_values('r'))

    colors = ['#2980b9' if v < 0 else '#e74c3c' for v in sub['r']]
    bars = ax.barh(sub['Input'], sub['r'], color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.4, height=0.6)

    for bar, val in zip(bars, sub['r']):
        x_pos = val + (0.02 if val >= 0 else -0.02)
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.2f}', va='center', ha=ha,
                fontsize=ANNOT_SZ + 1, fontweight='bold', color='#222222')

    ax.axvline(0, color='#555555', lw=1.2)
    ax.set_xlim(-1.15, 1.15)
    ax.set_title(out_label, **TITLE_KW)
    ax.set_xlabel("Spearman's ρ", **LABEL_KW)
    ax.set_facecolor('#f5f6fa')
    ax.tick_params(axis='y', labelsize=TICK_SZ)
    ax.tick_params(axis='x', labelsize=TICK_SZ)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvspan( 0,    1.15, alpha=0.04, color='#e74c3c', zorder=0)
    ax.axvspan(-1.15, 0,    alpha=0.04, color='#2980b9', zorder=0)

legend_elements = [
    Patch(facecolor='#e74c3c', alpha=0.85, label='Positive correlation'),
    Patch(facecolor='#2980b9', alpha=0.85, label='Negative correlation'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           fontsize=LEGEND_SZ + 1, framealpha=0.6, bbox_to_anchor=(0.5, -0.03))

fig.suptitle("Sensitivity Analysis — Spearman Rank Correlation  ·  Top 10 Drivers per Output",
             fontsize=17, fontweight='bold', color='#111111')
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig('Plot3_Tornado.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✓ Plot 3 saved")


# ══════════════════════════════════════════════════════════
# PLOT 4 — Parallel Coordinates, colored by L/D
# ══════════════════════════════════════════════════════════
import matplotlib.patches as mpatches

pc_cols = {
    'Wingspan':      'Wingspan (m)',
    'AR':            'Aspect Ratio',
    'Sweeps':        'Inner Sweep (°)',
    'Root_Chords':   'Chord (m)',
    'Stress_Ratio':  'Stress Ratio',
    'Static_Margin': 'Static Margin',
    'Cost_Per_Hour': 'Cost/seat-mile',
    'L_D_Cruise':    'L/D',
}
pc_keys  = [k for k in pc_cols if k in df.columns]
pc_names = [pc_cols[k] for k in pc_keys]

pc_data = df[pc_keys].copy().dropna()
pc_norm = (pc_data - pc_data.min()) / (pc_data.max() - pc_data.min())

ld_vals = pc_data['L_D_Cruise'].values
ld_norm = mcolors.Normalize(vmin=ld_vals.min(), vmax=ld_vals.max())
ld_cmap = plt.cm.RdYlGn

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f5f6fa')

n_axes = len(pc_keys)
x_pos  = list(range(n_axes))

for i, row in pc_norm.iterrows():
    color = ld_cmap(ld_norm(ld_vals[pc_data.index.get_loc(i)]))
    ax.plot(x_pos, row[pc_keys].values, color=color, alpha=0.15, lw=0.7)

top_mask = ld_vals >= np.percentile(ld_vals, 98)
for i, row in pc_norm[top_mask].iterrows():
    color = ld_cmap(ld_norm(ld_vals[pc_data.index.get_loc(i)]))
    ax.plot(x_pos, row[pc_keys].values, color=color, alpha=0.7, lw=1.6)

for xi, (key, name) in enumerate(zip(pc_keys, pc_names)):
    ax.axvline(xi, color='#888888', lw=1.0, zorder=3)
    raw_min = pc_data[key].min()
    raw_max = pc_data[key].max()
    fmt = '{:.0f}' if abs(raw_max) > 10 else '{:.2f}'
    ax.text(xi, -0.05, fmt.format(raw_min), ha='center', va='top',
            fontsize=TICK_SZ - 2, color='#555555')
    ax.text(xi,  1.05, fmt.format(raw_max), ha='center', va='bottom',
            fontsize=TICK_SZ - 2, color='#555555')

ax.set_xticks(x_pos)
ax.set_xticklabels(pc_names, fontsize=TICK_SZ, rotation=15, ha='right')
ax.set_yticks([])
ax.set_xlim(-0.3, n_axes - 0.7)
ax.set_ylim(-0.15, 1.15)
for spine in ax.spines.values():
    spine.set_visible(False)

sm = cm.ScalarMappable(cmap=ld_cmap, norm=ld_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.025, shrink=0.7)
cbar.set_label('L/D Cruise', fontsize=CBAR_SZ)
cbar.ax.yaxis.set_tick_params(color='#444444')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#444444', fontsize=TICK_SZ - 1)

top_patch = mpatches.Patch(color=ld_cmap(0.95), alpha=0.8, label='Top 2% L/D highlighted')
ax.legend(handles=[top_patch], fontsize=LEGEND_SZ, framealpha=0.6, loc='upper left')

ax.set_title('Parallel Coordinates — Design Traces Colored by L/D', **TITLE_KW)
plt.tight_layout()
plt.savefig('Plot4_Parallel_Coords.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✓ Plot 4 saved")


# ══════════════════════════════════════════════════════════
# PLOT 5 — L/D vs MTOW, colored by Static Margin
# ══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('white')

sm_vals = df['Static_Margin'].dropna()
sm_norm = mcolors.Normalize(vmin=sm_vals.min(), vmax=sm_vals.max())
sm_cmap = mcolors.LinearSegmentedColormap.from_list(
    'sm_cmap',
    [(0.0, '#e74c3c'), (0.5, '#f1c40f'), (1.0, '#2980b9')],
    N=512
)

sc = ax.scatter(
    df['MTOW'] / 1e6, df['L_D_Cruise'],
    c=df['Static_Margin'], cmap=sm_cmap, norm=sm_norm,
    s=35, alpha=0.75, edgecolors='none'
)

sm_mappable = cm.ScalarMappable(cmap=sm_cmap, norm=sm_norm)
sm_mappable.set_array([])
cbar = plt.colorbar(sm_mappable, ax=ax, pad=0.02, fraction=0.035)
cbar.set_label('Static Margin', fontsize=CBAR_SZ)
cbar.ax.yaxis.set_tick_params(color='#444444')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#444444', fontsize=TICK_SZ - 1)

ax.set_title('L/D vs MTOW  ·  Colored by Static Margin', **TITLE_KW)
ax.set_xlabel('MTOW (MN)', **LABEL_KW)
ax.set_ylabel('Lift-to-Drag Ratio (L/D)', **LABEL_KW)
ax.tick_params(axis='both', labelsize=TICK_SZ)
ax.set_facecolor('#f5f6fa')
plt.tight_layout()
plt.savefig('Plot5_LD_vs_MTOW.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✓ Plot 5 saved")


# ══════════════════════════════════════════════════════════
# PLOT 6 — Stress Feasibility Map
# ══════════════════════════════════════════════════════════
from scipy.interpolate import griddata

fig, ax = plt.subplots(figsize=(13, 8))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f5f6fa')

sub = df[['Wingspan', 'Root_Chords', 'Stress_Ratio', 'L_D_Cruise']].dropna()

pass_mask = sub['Stress_Ratio'] <= 0.9
fail_mask = ~pass_mask

ax.scatter(sub.loc[pass_mask, 'Wingspan'], sub.loc[pass_mask, 'Root_Chords'],
           c='#2ecc71', alpha=0.18, s=20, edgecolors='none', label='Feasible (SR ≤ 0.9)')
ax.scatter(sub.loc[fail_mask, 'Wingspan'], sub.loc[fail_mask, 'Root_Chords'],
           c='#e74c3c', alpha=0.18, s=20, edgecolors='none', label='Overstressed (SR > 0.9)')

grid_x = np.linspace(sub['Wingspan'].min(), sub['Wingspan'].max(), 200)
grid_y = np.linspace(sub['Root_Chords'].min(), sub['Root_Chords'].max(), 200)
gx, gy = np.meshgrid(grid_x, grid_y)
gz_ld  = griddata((sub['Wingspan'], sub['Root_Chords']), sub['L_D_Cruise'],
                  (gx, gy), method='linear')

contour_f = ax.contourf(gx, gy, gz_ld, levels=10, cmap='Blues', alpha=0.35, zorder=2)
contour_l = ax.contour( gx, gy, gz_ld, levels=10, colors='#2c3e50',
                        linewidths=0.8, alpha=0.6, zorder=3)
ax.clabel(contour_l, fmt='%.1f', fontsize=TICK_SZ - 1, colors='#2c3e50')

gz_sr = griddata((sub['Wingspan'], sub['Root_Chords']), sub['Stress_Ratio'],
                 (gx, gy), method='linear')
ax.contour(gx, gy, gz_sr, levels=[0.9], colors='#e74c3c',
           linewidths=2.2, linestyles='--', zorder=4)
ax.plot([], [], color='#e74c3c', lw=2.2, ls='--', label='Structural limit (SR = 1.0)')

ax.axvline(64.0, color='#2980b9', lw=1.6, ls=':', alpha=0.85, label='Code E Gate (65 m)')

cbar = plt.colorbar(contour_f, ax=ax, pad=0.02, fraction=0.035)
cbar.set_label('L/D Cruise', fontsize=CBAR_SZ)
cbar.ax.yaxis.set_tick_params(color='#444444')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#444444', fontsize=TICK_SZ - 1)

ax.set_title('Structural Feasibility Map — Contours = L/D Cruise', **TITLE_KW)
ax.set_xlabel('Wingspan (m)', **LABEL_KW)
ax.set_ylabel('Root Chord (m)', **LABEL_KW)
ax.tick_params(axis='both', labelsize=TICK_SZ)
ax.legend(fontsize=LEGEND_SZ, framealpha=0.6, loc='upper left')
plt.tight_layout()
plt.savefig('Plot6_Feasibility_Map.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✓ Plot 6 saved")

print("\n✅ All 6 plots saved.")