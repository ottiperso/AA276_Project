# plot_representative.py
# individual IC trajectory plots in both BRT slices

import numpy as np
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

os.makedirs('outputs/plots', exist_ok=True)

values_converged = np.load('outputs/data/values.npy')[-1]
GRID_RESOLUTION = (15, 15, 15, 15, 15, 15)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(np.full(6,-8.), np.full(6,8.)), GRID_RESOLUTION)
interp = RegularGridInterpolator(
    [np.array(v) for v in grid.coordinate_vectors],
    np.array(values_converged), bounds_error=False, fill_value=None)

# BRT slices
dpz = np.linspace(-8, 8, 100)
dvz = np.linspace(-8, 8, 100)
dpx = np.linspace(-8, 8, 100)

DPZ_v, DVZ = np.meshgrid(dpz, dvz)
pts_vz = np.stack([np.zeros_like(DPZ_v.ravel()), np.zeros_like(DPZ_v.ravel()),
    DPZ_v.ravel(), np.zeros_like(DPZ_v.ravel()), np.zeros_like(DPZ_v.ravel()),
    DVZ.ravel()], axis=1)
V_pzvz = interp(pts_vz).reshape(DPZ_v.shape)

DPX_p, DPZ_p = np.meshgrid(dpx, dpz)
pts_px = np.stack([DPX_p.ravel(), np.zeros_like(DPX_p.ravel()),
    DPZ_p.ravel(), np.zeros_like(DPX_p.ravel()), np.zeros_like(DPX_p.ravel()),
    np.zeros_like(DPX_p.ravel())], axis=1)
V_pxpz = interp(pts_px).reshape(DPX_p.shape)

# ICS = {
#     'inside_brt': ('darkgreen', '$V=-1.05$, Captured @ 0.33s'),
#     'boundary_diag': ('orange', '$V=+0.005$, Escaped'),
#     'outside_near': ('red', '$V=+2.13$, Escaped'),
# }

ICS = {
    'inside_brt': ('darkgreen', '$V=-1.05$, Captured @ 0.33s'),
    'inside_far': ('limegreen', '$V=-0.75$, Captured'),
    'boundary_diag': ('orange', '$V=+0.005$, Escaped'),
    'boundary_pz_other': ('goldenrod', '$V=+0.003$, Captured*'),
    'boundary_dvz_other': ('darkorange', '$V=+0.009$, Escaped'),
    'outside_near': ('red', '$V=+2.13$, Escaped'),
}

for ic_name, (color, subtitle) in ICS.items():
    zs = np.load(f'outputs/data/zs_{ic_name}.npy')
    dt = 0.01
    t = np.arange(len(zs)) * dt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # panel 1: pz vs vz
    ax = axes[0]
    ax.pcolormesh(dpz, dvz, V_pzvz, cmap='RdBu', shading='auto', vmin=-3, vmax=3, alpha=0.7)
    ax.contour(dpz, dvz, V_pzvz, levels=[0], colors='k', linewidths=2)
    ax.contourf(dpz, dvz, V_pzvz, levels=[V_pzvz.min(), 0], colors=['red'], alpha=0.15)
    ax.axvline(-1.0, color='g', linestyle='--', linewidth=1.5, label='Capture radius')
    ax.axvline( 1.0, color='g', linestyle='--', linewidth=1.5)
    ax.plot(zs[:, 2], zs[:, 5], color=color, linewidth=2.5)
    ax.plot(zs[0, 2], zs[0, 5], 'o', color=color, markersize=10, markeredgecolor='k', label='Start')
    ax.plot(zs[-1, 2], zs[-1, 5], 's', color=color, markersize=10, markeredgecolor='k', label='End')
    ax.set_xlabel(r'$\Delta p_z$ (m)', fontsize=12)
    ax.set_ylabel(r'$\Delta v_z$ (m/s)', fontsize=12)
    ax.set_title(r'$\Delta p_z$ vs $\Delta v_z$ phase plane', fontsize=11)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # panel 2: px vs pz
    ax = axes[1]
    ax.pcolormesh(dpx, dpz, V_pxpz, cmap='RdBu', shading='auto', vmin=-3, vmax=3, alpha=0.7)
    ax.contour(dpx, dpz, V_pxpz, levels=[0], colors='k', linewidths=2)
    ax.contourf(dpx, dpz, V_pxpz, levels=[V_pxpz.min(), 0], colors=['red'], alpha=0.15)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'g--', linewidth=1.5, label='Capture radius')
    ax.plot(zs[:, 0], zs[:, 2], color=color, linewidth=2.5)
    ax.plot(zs[0, 0], zs[0, 2], 'o', color=color, markersize=10, markeredgecolor='k', label='Start')
    ax.plot(zs[-1, 0], zs[-1, 2], 's', color=color, markersize=10, markeredgecolor='k', label='End')
    ax.set_xlabel(r'$\Delta p_x$ (m)', fontsize=12)
    ax.set_ylabel(r'$\Delta p_z$ (m)', fontsize=12)
    ax.set_title(r'$\Delta p_x$ vs $\Delta p_z$ position plane', fontsize=11)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{ic_name} — {subtitle}', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'outputs/plots/rep_trajectory_{ic_name}.png', bbox_inches='tight', dpi=150)
    print(f'Saved rep_trajectory_{ic_name}.png')