import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
matplotlib.use('Agg')

from dynamics import PursuitEvasion, F_P_MAX, F_E_MAX

values_converged = np.load('outputs/data/values.npy')[-1]

# load all initial conditions
IC_NAMES = ['inside_brt', 'inside_far', 'boundary', 'outside_near', 'outside_far']

results = {}
for name in IC_NAMES:
    results[name] = dict(
        zs  = np.load(f'outputs/data/zs_{name}.npy'),
        p_P = np.load(f'outputs/data/p_P_{name}.npy'),
        p_E = np.load(f'outputs/data/p_E_{name}.npy'),
        FPs = np.load(f'outputs/data/FPs_{name}.npy'),
        FEs = np.load(f'outputs/data/FEs_{name}.npy'),
    )

GRID_RESOLUTION = (15, 15, 15, 15, 15, 15)
# GRID_RESOLUTION = (21, 21, 21, 21, 21, 21)
# GRID_RESOLUTION = (16, 16, 16, 16, 16, 16)

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-8., -8., -8., -8., -8., -8.]),
        np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
    ),
    # hj.sets.Box(
    #     np.array([-10., -10., -10., -10., -10., -10.]),  
    #     np.array([ 10.,  10.,  10.,  10.,  10.,  10.])
    # ),
    GRID_RESOLUTION
)

values_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(values_converged),
    bounds_error=False,
    fill_value=None
)

# color + style per IC
IC_COLORS = {
    'inside_brt':  'darkgreen',
    'inside_far':   'limegreen',
    'boundary':     'orange',
    'outside_near': 'magenta',
    'outside_far':  'red',
}

# 2D slice: delta_pz (x-axis) vs delta_vz (y-axis)
dpz = np.linspace(-8, 8, 101)
dvz = np.linspace(-8, 8, 101)
# dpz = np.linspace(-10, 10, 101)
# dvz = np.linspace(-10, 10, 101)

DPZ, DVZ = np.meshgrid(dpz, dvz)
slice_pts = np.stack([
    np.zeros_like(DPZ.ravel()),  # delta_px = 0
    np.zeros_like(DPZ.ravel()),  # delta_py = 0
    DPZ.ravel(),                 # delta_pz (x)
    np.zeros_like(DPZ.ravel()),  # delta_vx = 0
    np.zeros_like(DPZ.ravel()),  # delta_vy = 0
    DVZ.ravel(),                 # delta_vz (y)
], axis=1)
V_slice = values_converged_interpolator(slice_pts).reshape(DVZ.shape)

dt = 0.01

# ── Figure 1: BRT slice + relative distance + 5 control profile subplots ──
fig = plt.figure(figsize=(24, 8))
gs = GridSpec(1, 3, figure=fig, wspace=0.35)

ax_brt  = fig.add_subplot(gs[0])
ax_dist = fig.add_subplot(gs[1])
gs_ctrl = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[2], hspace=0.7)
ctrl_axes = [fig.add_subplot(gs_ctrl[i]) for i in range(5)]

# plot 1: BRT slice + all trajectories overlaid
ax = ax_brt
ax.pcolormesh(dpz, dvz, V_slice, cmap='RdBu', vmin=-3, vmax=3)
ax.contour(dpz, dvz, V_slice, levels=[0], colors='k', linewidths=2)
for name in IC_NAMES:
    zs = results[name]['zs']
    ax.plot(zs[:, 2], zs[:, 5], color=IC_COLORS[name], linewidth=2, label=name)
ax.axvline(-1.0, color='r', linestyle='--', linewidth=1.5, label='Capture radius')
ax.axvline( 1.0, color='r', linestyle='--', linewidth=1.5)
ax.set_xlabel(r'$\Delta p_z$ (m)')
ax.set_ylabel(r'$\Delta v_z$ (m/s)')
ax.set_title(r'BRT slice ($\Delta p_x=\Delta p_y=0$, $\Delta v_x=\Delta v_y=0$)')
ax.legend(fontsize=7)

# plot 2: relative distance over time for all ICs
ax = ax_dist
for name in IC_NAMES:
    zs = results[name]['zs']
    t = np.arange(len(zs)) * dt
    dist = np.sqrt(zs[:,0]**2 + zs[:,1]**2 + zs[:,2]**2)
    ax.plot(t, dist, color=IC_COLORS[name], linewidth=2, label=name)
ax.axhline(1.0, color='r', linestyle='--', linewidth=1.5, label='Capture radius')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\|\Delta p\|$ (m)')
ax.set_title('Relative distance over time')
ax.legend(fontsize=7)
ax.grid(True)

# plot 3: 5 separate control profile subplots
for i, name in enumerate(IC_NAMES):
    ax = ctrl_axes[i]
    FPs = results[name]['FPs']
    FEs = results[name]['FEs']
    t_ctrl = np.arange(len(FPs)) * dt
    ax.plot(t_ctrl, FPs, color=IC_COLORS[name], linewidth=1.2, label='Pursuer')
    ax.plot(t_ctrl, FEs, color=IC_COLORS[name], linewidth=1.2, linestyle='--', label='Evader')
    ax.set_ylabel('N', fontsize=7)
    ax.set_title(name, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True)
    ax.legend(fontsize=6)
    if i < len(IC_NAMES) - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time (s)', fontsize=7)

fig.savefig('outputs/plots/pursuit_evasion.png', bbox_inches='tight')
print('Saved plot to outputs/plots/pursuit_evasion.png')

# ── Figure 2: per-IC detail plots ──
for name in IC_NAMES:
    zs  = results[name]['zs']
    p_P = results[name]['p_P']
    p_E = results[name]['p_E']
    FPs = results[name]['FPs']
    FEs = results[name]['FEs']

    t      = np.arange(len(zs)) * dt
    t_ctrl = np.arange(len(FPs)) * dt

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # plot 1: individual drone positions
    ax = axes2[0]
    ax.plot(t, p_P[:, 2], 'g', linewidth=2, label='Pursuer $p_z$')
    ax.plot(t, p_E[:, 2], 'm', linewidth=2, label='Evader $p_z$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$p_z$ (m)')
    ax.set_title('Individual $p_z$ positions over time')
    ax.legend()
    ax.grid(True)

    # plot 2: relative state over time
    ax = axes2[1]
    ax.plot(t, zs[:, 2], 'b', linewidth=2, label=r'$\Delta p_z$')
    ax.plot(t, zs[:, 5], 'r', linewidth=2, label=r'$\Delta v_z$')
    ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_title('Relative state over time')
    ax.legend()
    ax.grid(True)

    # plot 3: control inputs
    ax = axes2[2]
    ax.plot(t_ctrl, FPs, 'g', linewidth=2, label='Pursuer $F_P$')
    ax.plot(t_ctrl, FEs, 'm', linewidth=2, label='Evader $F_E$')
    ax.axhline( F_P_MAX, color='g', linestyle=':', linewidth=0.8, label='$F_P$ max')
    ax.axhline(-F_P_MAX, color='g', linestyle=':', linewidth=0.8)
    ax.axhline( F_E_MAX, color='m', linestyle=':', linewidth=0.8, label='$F_E$ max')
    ax.axhline(-F_E_MAX, color='m', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Thrust (N)')
    ax.set_title('Control inputs (bang-bang)')
    ax.legend()
    ax.grid(True)

    fig2.tight_layout()
    fig2.savefig(f'outputs/plots/trajectories_{name}.png')
    print(f'Saved plot to outputs/plots/trajectories_{name}.png')


# ── Figure 3: two-panel BRT only ──
def plot_brt_only(values_converged_interpolator):
    """
    Two-panel BRT plot:
      Left:  delta_px (x) vs delta_pz (y), slice at delta_py=0, delta_v=0
      Right: delta_pz (x) vs delta_vz (y), slice at delta_px=delta_py=0, delta_vx=delta_vy=0
    """
    dpx = np.linspace(-8, 8, 201)
    dpz = np.linspace(-8, 8, 201)
    dvz = np.linspace(-8, 8, 201)

    # dpx = np.linspace(-10, 10, 201)
    # dpz = np.linspace(-10, 10, 201)
    # dvz = np.linspace(-10, 10, 201)

    # Panel 1: delta_px (x) vs delta_pz (y)
    DPX, DPZ_pos = np.meshgrid(dpx, dpz)
    slice_pos = np.stack([
        DPX.ravel(),
        np.zeros_like(DPX.ravel()),
        DPZ_pos.ravel(),
        np.zeros_like(DPX.ravel()),
        np.zeros_like(DPX.ravel()),
        np.zeros_like(DPX.ravel()),
    ], axis=1)
    V_pos = values_converged_interpolator(slice_pos).reshape(DPX.shape)

    # Panel 2: delta_pz (x) vs delta_vz (y)
    DPZ_vel, DVZ = np.meshgrid(dpz, dvz)
    slice_vel = np.stack([
        np.zeros_like(DPZ_vel.ravel()),
        np.zeros_like(DPZ_vel.ravel()),
        DPZ_vel.ravel(),
        np.zeros_like(DPZ_vel.ravel()),
        np.zeros_like(DPZ_vel.ravel()),
        DVZ.ravel(),
    ], axis=1)
    V_vel = values_converged_interpolator(slice_vel).reshape(DPZ_vel.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1
    ax = axes[0]
    pcm1 = ax.pcolormesh(dpx, dpz, V_pos, cmap='RdBu', shading='auto')
    ax.contour(dpx, dpz, V_pos, levels=[0], colors='k', linewidths=2)
    ax.contourf(dpx, dpz, V_pos, levels=[V_pos.min(), 0], colors=['red'], alpha=0.3)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'g--', linewidth=2, label='Capture radius')
    ax.set_xlabel(r'$\Delta p_x$ (m)')
    ax.set_ylabel(r'$\Delta p_z$ (m)')
    ax.set_title(r'BRT: $\Delta p_x$ vs $\Delta p_z$' '\n' r'($\Delta p_y=0$, $\Delta v=0$)')
    ax.set_aspect('equal')
    plt.colorbar(pcm1, ax=ax, label='Value Function V')
    ax.legend()

    # Panel 2
    ax = axes[1]
    pcm2 = ax.pcolormesh(dpz, dvz, V_vel, cmap='RdBu', shading='auto')
    ax.contour(dpz, dvz, V_vel, levels=[0], colors='k', linewidths=2)
    ax.contourf(dpz, dvz, V_vel, levels=[V_vel.min(), 0], colors=['red'], alpha=0.3)
    ax.axvline(-1.0, color='g', linestyle='--', linewidth=2, label='Capture radius')
    ax.axvline( 1.0, color='g', linestyle='--', linewidth=2)
    ax.set_xlabel(r'$\Delta p_z$ (m)')
    ax.set_ylabel(r'$\Delta v_z$ (m/s)')
    ax.set_title(r'BRT: $\Delta p_z$ vs $\Delta v_z$' '\n'
                 r'($\Delta p_x=\Delta p_y=0$, $\Delta v_x=\Delta v_y=0$)')
    plt.colorbar(pcm2, ax=ax, label='Value Function V')
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig('outputs/plots/brt_only.png')
    print('Saved BRT-only plot to outputs/plots/brt_only.png')


def plot_brt_over_time(values_all, times, values_converged_interpolator_fn):
    """
    Plot BRT slices at several timesteps to show growth over time.
    8 timesteps arranged as 4 rows x 4 cols:
      row 0: px vs pz, first 4 timesteps
      row 1: pz vs vz, first 4 timesteps
      row 2: px vs pz, last 4 timesteps
      row 3: pz vs vz, last 4 timesteps
    """
    n_times = len(times)
    indices = [0, n_times//10, n_times//8, n_times//6, n_times//4, n_times//3, n_times//2, n_times-1]
    selected_times = [times[i] for i in indices]

    dpx = np.linspace(-8, 8, 201)
    dpz = np.linspace(-8, 8, 201)
    dvz = np.linspace(-8, 8, 201)

    DPX, DPZ_pos = np.meshgrid(dpx, dpz)
    slice_pos_pts = np.stack([
        DPX.ravel(), np.zeros_like(DPX.ravel()), DPZ_pos.ravel(),
        np.zeros_like(DPX.ravel()), np.zeros_like(DPX.ravel()), np.zeros_like(DPX.ravel()),
    ], axis=1)

    DPZ_vel, DVZ = np.meshgrid(dpz, dvz)
    slice_vel_pts = np.stack([
        np.zeros_like(DPZ_vel.ravel()), np.zeros_like(DPZ_vel.ravel()), DPZ_vel.ravel(),
        np.zeros_like(DPZ_vel.ravel()), np.zeros_like(DPZ_vel.ravel()), DVZ.ravel(),
    ], axis=1)

    n_cols = len(indices) // 2  # 4 columns
    fig, axes = plt.subplots(4, n_cols, figsize=(5*n_cols, 20))

    for col, (idx, t) in enumerate(zip(indices, selected_times)):
        interp_t = RegularGridInterpolator(
            ([np.array(v) for v in grid.coordinate_vectors]),
            np.array(values_all[idx]),
            bounds_error=False, fill_value=None
        )

        V_pos = interp_t(slice_pos_pts).reshape(DPX.shape)
        V_vel = interp_t(slice_vel_pts).reshape(DPZ_vel.shape)

        # first 4 timesteps go in rows 0-1, last 4 in rows 2-3
        row_base = (col // n_cols) * 2
        col_idx  = col % n_cols

        # px vs pz
        ax = axes[col // n_cols, col_idx]
        ax.pcolormesh(dpx, dpz, V_pos, cmap='RdBu', shading='auto')
        ax.contour(dpx, dpz, V_pos, levels=[0], colors='k', linewidths=2)
        ax.contourf(dpx, dpz, V_pos, levels=[V_pos.min(), 0], colors=['red'], alpha=0.3)
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'g--', linewidth=1.5)
        title_str = f't = {t:.1f}s'
        if col == 0:
            title_str += ' (initial)'
        if col == len(indices) - 1:
            title_str += ' (converged)'
        ax.set_title(title_str, fontsize=11)
        ax.set_xlabel(r'$\Delta p_x$ (m)')
        if col_idx == 0:
            ax.set_ylabel(r'$\Delta p_z$ (m)')
        ax.set_aspect('equal')

        # pz vs vz
        ax = axes[2 + col // n_cols, col_idx]
        ax.pcolormesh(dpz, dvz, V_vel, cmap='RdBu', shading='auto')
        ax.contour(dpz, dvz, V_vel, levels=[0], colors='k', linewidths=2)
        ax.contourf(dpz, dvz, V_vel, levels=[V_vel.min(), 0], colors=['red'], alpha=0.3)
        ax.axvline(-1.0, color='g', linestyle='--', linewidth=1.5)
        ax.axvline( 1.0, color='g', linestyle='--', linewidth=1.5)
        ax.set_xlabel(r'$\Delta p_z$ (m)')
        if col_idx == 0:
            ax.set_ylabel(r'$\Delta v_z$ (m/s)')

    fig.suptitle('BRT Evolution Over Time', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig('outputs/plots/brt_over_time.png', bbox_inches='tight')
    print('Saved outputs/plots/brt_over_time.png')


plot_brt_only(values_converged_interpolator)
values_all = np.load('outputs/data/values.npy')
times_all  = np.load('outputs/data/times.npy')
plot_brt_over_time(values_all, times_all, values_converged_interpolator)