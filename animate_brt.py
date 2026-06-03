"""
BRT Animation Script
--------------------
Produces two animations per IC:
  1. pz vs vz phase plane
  2. px vs pz position plane

BRT background updates each frame: at trajectory time t, shows the BRT
computed for time horizon t (so BRT grows as trajectory plays forward).

Usage:
  python animate_brt.py --ic inside_brt
  python animate_brt.py --ic all
  python animate_brt.py --ic all --skip 3 --fps 30
"""

import numpy as np
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
parser.add_argument('--ic',   type=str, default='inside_brt',
                    help='IC name or "all"')
parser.add_argument('--fps',  type=int, default=30,
                    help='Frames per second')
parser.add_argument('--skip', type=int, default=5,
                    help='Plot every N trajectory steps')
args = parser.parse_args()

# ── load BRT ──
print('Loading BRT data...')
values_all = np.load('outputs/data/values.npy')   # (n_times, 15,15,15,15,15,15)
times_all  = np.load('outputs/data/times.npy')    # (n_times,) negative, 0 to -20

GRID_RESOLUTION = (15, 15, 15, 15, 15, 15)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-8., -8., -8., -8., -8., -8.]),
        np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
    ),
    GRID_RESOLUTION
)

# ── precompute slice grids ──
dpz = np.linspace(-8, 8, 100)
dvz = np.linspace(-8, 8, 100)
dpx = np.linspace(-8, 8, 100)

# slice 1: pz (x) vs vz (y)
DPZ_s1, DVZ_s1 = np.meshgrid(dpz, dvz)
pts_s1 = np.stack([
    np.zeros_like(DPZ_s1.ravel()),
    np.zeros_like(DPZ_s1.ravel()),
    DPZ_s1.ravel(),
    np.zeros_like(DPZ_s1.ravel()),
    np.zeros_like(DPZ_s1.ravel()),
    DVZ_s1.ravel(),
], axis=1)

# slice 2: px (x) vs pz (y)
DPX_s2, DPZ_s2 = np.meshgrid(dpx, dpz)
pts_s2 = np.stack([
    DPX_s2.ravel(),
    np.zeros_like(DPX_s2.ravel()),
    DPZ_s2.ravel(),
    np.zeros_like(DPX_s2.ravel()),
    np.zeros_like(DPX_s2.ravel()),
    np.zeros_like(DPX_s2.ravel()),
], axis=1)

# ── precompute V slices for ALL timesteps ──
print(f'Precomputing BRT slices for {len(times_all)} timesteps...')
V_s1_all = np.zeros((len(times_all), len(dvz), len(dpz)))
V_s2_all = np.zeros((len(times_all), len(dpz), len(dpx)))

for t_idx in range(len(times_all)):
    interp_t = RegularGridInterpolator(
        ([np.array(v) for v in grid.coordinate_vectors]),
        np.array(values_all[t_idx]),
        bounds_error=False, fill_value=None
    )
    V_s1_all[t_idx] = interp_t(pts_s1).reshape(DPZ_s1.shape)
    V_s2_all[t_idx] = interp_t(pts_s2).reshape(DPX_s2.shape)
    if t_idx % 20 == 0:
        print(f'  {t_idx}/{len(times_all)}')

print('Done precomputing.')

IC_NAMES = ['inside_brt', 'inside_far', 'boundary', 'outside_near', 'outside_far']
IC_COLORS = {
    'inside_brt':   'limegreen',
    'inside_far':   'green',
    'boundary':     'orange',
    'outside_near': 'magenta',
    'outside_far':  'red',
}

ics_to_run = IC_NAMES if args.ic == 'all' else [args.ic]


def make_animation(ic_name):
    zs = np.load(f'outputs/data/zs_{ic_name}.npy')
    dt = 0.01
    color = IC_COLORS.get(ic_name, 'cyan')

    frames_idx = list(range(0, len(zs), args.skip))
    if frames_idx[-1] != len(zs) - 1:
        frames_idx.append(len(zs) - 1)
    zs_frames  = zs[frames_idx]
    times_traj = np.array(frames_idx) * dt

    # map each trajectory frame to nearest BRT timestep
    # at traj t, show BRT for horizon = -t (small at start, grows as t increases)
    brt_indices = []
    for traj_t in times_traj:
        target_brt_t = -traj_t  # t=0 -> brt t=0 (capture set), t=8 -> brt t=-8
        brt_idx = int(np.argmin(np.abs(times_all - target_brt_t)))
        brt_indices.append(brt_idx)

    theta = np.linspace(0, 2*np.pi, 200)

    # ── Animation 1: pz vs vz ──
    fig1, ax1 = plt.subplots(figsize=(7, 6))

    pcm1 = ax1.pcolormesh(dpz, dvz, V_s1_all[brt_indices[0]],
                          cmap='RdBu', shading='auto', vmin=-3, vmax=3, alpha=0.85)
    plt.colorbar(pcm1, ax=ax1, label='Value Function $V$')
    ax1.set_xlabel(r'$\Delta p_z$ (m)', fontsize=13)
    ax1.set_ylabel(r'$\Delta v_z$ (m/s)', fontsize=13)
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-8, 8)

    trail1, = ax1.plot([], [], '-', color=color, alpha=0.6, linewidth=1.5)
    dot1,   = ax1.plot([], [], 'o', color=color, markersize=10,
                       markeredgecolor='k', markeredgewidth=1)
    title1  = ax1.set_title('')
    contour_artists1 = []

    def init1():
        trail1.set_data([], [])
        dot1.set_data([], [])
        return []

    def update1(i):
        nonlocal contour_artists1
        brt_idx = brt_indices[i]
        V = V_s1_all[brt_idx]

        # clear previous contours
        for c in contour_artists1:
            try:
                c.remove()
            except Exception:
                pass
        contour_artists1 = []

        # update heatmap
        pcm1.set_array(V.ravel())

        # redraw contours
        cs  = ax1.contour(dpz, dvz, V, levels=[0], colors='k', linewidths=2)
        cf  = ax1.contourf(dpz, dvz, V, levels=[V.min(), 0],
                           colors=['red'], alpha=0.15)
        vl1 = ax1.axvline(-1.0, color='g', linestyle='--', linewidth=1.5)
        vl2 = ax1.axvline( 1.0, color='g', linestyle='--', linewidth=1.5)
        contour_artists1 = list(cs.collections) + list(cf.collections) + [vl1, vl2]

        trail1.set_data(zs_frames[:i+1, 2], zs_frames[:i+1, 5])
        dot1.set_data([zs_frames[i, 2]], [zs_frames[i, 5]])

        brt_t  = times_all[brt_idx]
        traj_t = times_traj[i]
        dist   = np.sqrt(zs_frames[i,0]**2 + zs_frames[i,1]**2 + zs_frames[i,2]**2)
        title1.set_text(
            f'$\\Delta p_z$ vs $\\Delta v_z$  |  '
            f't = {traj_t:.2f}s  |  '
            f'BRT horizon = {brt_t:.1f}s  |  '
            f'dist = {dist:.2f} m'
        )
        return []

    ani1 = animation.FuncAnimation(
        fig1, update1, frames=len(frames_idx),
        init_func=init1, interval=1000//args.fps, blit=False
    )
    out1 = f'outputs/plots/animations/{ic_name}_pz_vz.mp4'
    ani1.save(out1, writer='ffmpeg', fps=args.fps, dpi=120)
    plt.close(fig1)
    print(f'Saved {out1}')

    # ── Animation 2: px vs pz ──
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    pcm2 = ax2.pcolormesh(dpx, dpz, V_s2_all[brt_indices[0]],
                          cmap='RdBu', shading='auto', vmin=-3, vmax=3, alpha=0.85)
    plt.colorbar(pcm2, ax=ax2, label='Value Function $V$')
    ax2.set_xlabel(r'$\Delta p_x$ (m)', fontsize=13)
    ax2.set_ylabel(r'$\Delta p_z$ (m)', fontsize=13)
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    ax2.set_aspect('equal')

    trail2, = ax2.plot([], [], '-', color=color, alpha=0.6, linewidth=1.5)
    dot2,   = ax2.plot([], [], 'o', color=color, markersize=10,
                       markeredgecolor='k', markeredgewidth=1)
    title2  = ax2.set_title('')
    contour_artists2 = []

    def init2():
        trail2.set_data([], [])
        dot2.set_data([], [])
        return []

    def update2(i):
        nonlocal contour_artists2
        brt_idx = brt_indices[i]
        V = V_s2_all[brt_idx]

        for c in contour_artists2:
            try:
                c.remove()
            except Exception:
                pass
        contour_artists2 = []

        pcm2.set_array(V.ravel())

        cs  = ax2.contour(dpx, dpz, V, levels=[0], colors='k', linewidths=2)
        cf  = ax2.contourf(dpx, dpz, V, levels=[V.min(), 0],
                           colors=['red'], alpha=0.15)
        cap, = ax2.plot(np.cos(theta), np.sin(theta), 'g--', linewidth=1.5)
        contour_artists2 = list(cs.collections) + list(cf.collections) + [cap]

        trail2.set_data(zs_frames[:i+1, 0], zs_frames[:i+1, 2])
        dot2.set_data([zs_frames[i, 0]], [zs_frames[i, 2]])

        brt_t  = times_all[brt_idx]
        traj_t = times_traj[i]
        dist   = np.sqrt(zs_frames[i,0]**2 + zs_frames[i,1]**2 + zs_frames[i,2]**2)
        title2.set_text(
            f'$\\Delta p_x$ vs $\\Delta p_z$  |  '
            f't = {traj_t:.2f}s  |  '
            f'BRT horizon = {brt_t:.1f}s  |  '
            f'dist = {dist:.2f} m'
        )
        return []

    ani2 = animation.FuncAnimation(
        fig2, update2, frames=len(frames_idx),
        init_func=init2, interval=1000//args.fps, blit=False
    )
    out2 = f'outputs/plots/animations/{ic_name}_px_pz.mp4'
    ani2.save(out2, writer='ffmpeg', fps=args.fps, dpi=120)
    plt.close(fig2)
    print(f'Saved {out2}')


for ic in ics_to_run:
    print(f'\n=== Animating {ic} ===')
    make_animation(ic)

print('\nDone! Check outputs/plots/animations/')