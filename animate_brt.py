"""
BRT Animation Script
--------------------
Produces two animations:
  1. pz vs vz phase plane (the dynamically meaningful slice)
  2. px vs pz position plane (the symmetric slice)

Both show the relative state trajectory as an animated dot with trail,
overlaid on the BRT value function heatmap and boundary contour.

Usage:
  python animate_brt.py --ic inside_brt
  python animate_brt.py --ic outside_near
  python animate_brt.py --ic all
"""

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use('Agg')
import argparse
import os

os.makedirs('outputs/plots/animations', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--ic', type=str, default='inside_brt',
                    help='IC name or "all"')
parser.add_argument('--fps', type=int, default=30,
                    help='Frames per second')
parser.add_argument('--skip', type=int, default=5,
                    help='Plot every N steps (speeds up animation)')
args = parser.parse_args()

# ── load BRT ──
values_converged = np.load('outputs/data/values.npy')[-1]

GRID_RESOLUTION = (15, 15, 15, 15, 15, 15)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-8., -8., -8., -8., -8., -8.]),
        np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
    ),
    GRID_RESOLUTION
)

values_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(values_converged),
    bounds_error=False,
    fill_value=None
)

# ── precompute BRT slices ──
dpz = np.linspace(-8, 8, 150)
dvz = np.linspace(-8, 8, 150)
dpx = np.linspace(-8, 8, 150)

# slice 1: pz (x-axis) vs vz (y-axis), px=py=vx=vy=0
DPZ_s1, DVZ_s1 = np.meshgrid(dpz, dvz)
pts_s1 = np.stack([
    np.zeros_like(DPZ_s1.ravel()),
    np.zeros_like(DPZ_s1.ravel()),
    DPZ_s1.ravel(),
    np.zeros_like(DPZ_s1.ravel()),
    np.zeros_like(DPZ_s1.ravel()),
    DVZ_s1.ravel(),
], axis=1)
V_s1 = values_converged_interpolator(pts_s1).reshape(DPZ_s1.shape)

# slice 2: px (x-axis) vs pz (y-axis), py=0, vx=vy=vz=0
DPX_s2, DPZ_s2 = np.meshgrid(dpx, dpz)
pts_s2 = np.stack([
    DPX_s2.ravel(),
    np.zeros_like(DPX_s2.ravel()),
    DPZ_s2.ravel(),
    np.zeros_like(DPX_s2.ravel()),
    np.zeros_like(DPX_s2.ravel()),
    np.zeros_like(DPX_s2.ravel()),
], axis=1)
V_s2 = values_converged_interpolator(pts_s2).reshape(DPX_s2.shape)

IC_NAMES = ['inside_brt', 'inside_far', 'boundary', 'outside_near', 'outside_far']
IC_COLORS = {
    'inside_brt':   'limegreen',
    'inside_far':   'green',
    'boundary':     'orange',
    'outside_near': 'magenta',
    'outside_far':  'red',
}

if args.ic == 'all':
    ics_to_run = IC_NAMES
else:
    ics_to_run = [args.ic]


def make_animation(ic_name):
    zs = np.load(f'outputs/data/zs_{ic_name}.npy')
    dt = 0.01
    color = IC_COLORS.get(ic_name, 'cyan')

    # subsample frames
    frames_idx = list(range(0, len(zs), args.skip))
    if frames_idx[-1] != len(zs) - 1:
        frames_idx.append(len(zs) - 1)
    zs_frames = zs[frames_idx]
    times = np.array(frames_idx) * dt

    # ── Animation 1: pz vs vz ──
    fig1, ax1 = plt.subplots(figsize=(7, 6))

    pcm1 = ax1.pcolormesh(dpz, dvz, V_s1, cmap='RdBu', shading='auto',
                          vmin=-3, vmax=3, alpha=0.85)
    ax1.contour(dpz, dvz, V_s1, levels=[0], colors='k', linewidths=2)
    ax1.contourf(dpz, dvz, V_s1, levels=[V_s1.min(), 0],
                 colors=['red'], alpha=0.15)
    ax1.axvline(-1.0, color='g', linestyle='--', linewidth=1.5,
                label='Capture radius')
    ax1.axvline( 1.0, color='g', linestyle='--', linewidth=1.5)
    plt.colorbar(pcm1, ax=ax1, label='Value Function $V$')
    ax1.set_xlabel(r'$\Delta p_z$ (m)', fontsize=13)
    ax1.set_ylabel(r'$\Delta v_z$ (m/s)', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)

    trail1,  = ax1.plot([], [], '-', color=color, alpha=0.5, linewidth=1.5)
    dot1,    = ax1.plot([], [], 'o', color=color, markersize=10,
                        markeredgecolor='k', markeredgewidth=1)
    title1   = ax1.set_title('')

    def init1():
        trail1.set_data([], [])
        dot1.set_data([], [])
        title1.set_text('')
        return trail1, dot1, title1

    def update1(i):
        z = zs_frames[i]
        t = times[i]
        # trail: history up to current frame
        trail1.set_data(zs_frames[:i+1, 2], zs_frames[:i+1, 5])
        dot1.set_data([z[2]], [z[5]])
        V_cur = values_converged_interpolator(z.reshape(1, -1)).item()
        title1.set_text(
            f'$\\Delta p_z$ vs $\\Delta v_z$  |  '
            f't = {t:.2f} s  |  '
            f'$V$ = {V_cur:.3f}  |  '
            f'dist = {np.sqrt(z[0]**2+z[1]**2+z[2]**2):.2f} m'
        )
        return trail1, dot1, title1

    ani1 = animation.FuncAnimation(
        fig1, update1, frames=len(frames_idx),
        init_func=init1, interval=1000//args.fps, blit=True
    )
    out1 = f'outputs/plots/animations/{ic_name}_pz_vz.mp4'
    ani1.save(out1, writer='ffmpeg', fps=args.fps, dpi=120)
    plt.close(fig1)
    print(f'Saved {out1}')

    # ── Animation 2: px vs pz ──
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    pcm2 = ax2.pcolormesh(dpx, dpz, V_s2, cmap='RdBu', shading='auto',
                          vmin=-3, vmax=3, alpha=0.85)
    ax2.contour(dpx, dpz, V_s2, levels=[0], colors='k', linewidths=2)
    ax2.contourf(dpx, dpz, V_s2, levels=[V_s2.min(), 0],
                 colors=['red'], alpha=0.15)
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), 'g--', linewidth=1.5,
             label='Capture radius')
    plt.colorbar(pcm2, ax=ax2, label='Value Function $V$')
    ax2.set_xlabel(r'$\Delta p_x$ (m)', fontsize=13)
    ax2.set_ylabel(r'$\Delta p_z$ (m)', fontsize=13)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    ax2.set_aspect('equal')

    trail2,  = ax2.plot([], [], '-', color=color, alpha=0.5, linewidth=1.5)
    dot2,    = ax2.plot([], [], 'o', color=color, markersize=10,
                        markeredgecolor='k', markeredgewidth=1)
    title2   = ax2.set_title('')

    def init2():
        trail2.set_data([], [])
        dot2.set_data([], [])
        title2.set_text('')
        return trail2, dot2, title2

    def update2(i):
        z = zs_frames[i]
        t = times[i]
        trail2.set_data(zs_frames[:i+1, 0], zs_frames[:i+1, 2])
        dot2.set_data([z[0]], [z[2]])
        V_cur = values_converged_interpolator(z.reshape(1, -1)).item()
        title2.set_text(
            f'$\\Delta p_x$ vs $\\Delta p_z$  |  '
            f't = {t:.2f} s  |  '
            f'$V$ = {V_cur:.3f}  |  '
            f'dist = {np.sqrt(z[0]**2+z[1]**2+z[2]**2):.2f} m'
        )
        return trail2, dot2, title2

    ani2 = animation.FuncAnimation(
        fig2, update2, frames=len(frames_idx),
        init_func=init2, interval=1000//args.fps, blit=True
    )
    out2 = f'outputs/plots/animations/{ic_name}_px_pz.mp4'
    ani2.save(out2, writer='ffmpeg', fps=args.fps, dpi=120)
    plt.close(fig2)
    print(f'Saved {out2}')


for ic in ics_to_run:
    print(f'\n=== Animating {ic} ===')
    make_animation(ic)

print('\nDone! Check outputs/plots/animations/')