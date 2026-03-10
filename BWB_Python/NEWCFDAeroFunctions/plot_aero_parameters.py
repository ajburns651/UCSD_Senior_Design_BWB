## === Script To Plot Aero Parameters === ##
import numpy as np
import matplotlib.pyplot as plt


def plot_aero(CL, CD, CL_star, CD_star, LD_max, geom, aoa_deg):

    # Compute L/D array
    LD = np.array(CL) / np.array(CD)   # element-wise

    # Find maximum L/D point
    max_LD_idx = np.argmax(LD)
    max_LD_val = LD[max_LD_idx]
    max_LD_aoa = aoa_deg[max_LD_idx]

    # Create figure with white background
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    fig.suptitle(f"BWB Performance Analysis (M = {geom.get('M', 0.85):.2f})",
                 fontsize=16, fontweight='bold', y=0.98)

    # ────────────────────────────────────────────────
    # Subplot 1: Drag Polar (CL vs CD)
    # ────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(CD, CL, linewidth=1.8, color='blue', label='Drag Polar')
    ax1.scatter(CD_star, CL_star, s=100, color='red', marker='o',
                edgecolor='black', zorder=10, label='Cruise Point')
    ax1.set_xlabel(r'$C_D$', fontsize=12)
    ax1.set_ylabel(r'$C_L$', fontsize=12)
    ax1.set_title('Total Drag Polar', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.tick_params(axis='both', labelsize=10)

    # ────────────────────────────────────────────────
    # Subplot 2: CL vs Alpha
    # ────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(aoa_deg, CL, linewidth=1.8, color='blue')
    ax2.axhline(y=CL_star, color='red', linestyle='--', linewidth=1.5,
                label=f'CL* = {CL_star:.4f}')
    ax2.set_xlabel(r'$\alpha$ (deg)', fontsize=12)
    ax2.set_ylabel(r'$C_L$', fontsize=12)
    ax2.set_title('Lift Curve', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=10)

    # ────────────────────────────────────────────────
    # Subplot 3: CD vs Alpha
    # ────────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(aoa_deg, CD, linewidth=1.8, color='blue')
    ax3.axhline(y=CD_star, color='red', linestyle='--', linewidth=1.5,
                label=f'CD* = {CD_star:.5f}')
    ax3.set_xlabel(r'$\alpha$ (deg)', fontsize=12)
    ax3.set_ylabel(r'$C_D$', fontsize=12)
    ax3.set_title('Total Drag Rise', fontsize=13)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left', fontsize=10)

    # ────────────────────────────────────────────────
    # Subplot 4: L/D vs Alpha
    # ────────────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(aoa_deg, LD, linewidth=1.8, color=(0, 0.5, 0), label='L/D')
    ax4.scatter(max_LD_aoa, max_LD_val, s=100, color='black', marker='o',
                edgecolor='white', zorder=10, label=f'Max L/D = {max_LD_val:.2f}')
    ax4.set_xlabel(r'$\alpha$ (deg)', fontsize=12)
    ax4.set_ylabel('L/D', fontsize=12)
    ax4.set_title('Aerodynamic Efficiency', fontsize=13)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='upper left', fontsize=10)

    # Adjust layout to prevent overlap with suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()