# import numpy as np
# import jax.numpy as jnp
# import hj_reachability as hj
# from scipy.interpolate import RegularGridInterpolator
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')

# from dynamics import PursuitEvasion, F_P_MAX, F_E_MAX

# values_converged = np.load('outputs/data/values.npy')[-1]
# zs_in   = np.load('outputs/data/zs_inside.npy')
# zs_out  = np.load('outputs/data/zs_outside.npy')
# p_P_in  = np.load('outputs/data/p_P_inside.npy')
# p_P_out = np.load('outputs/data/p_P_outside.npy')
# p_E_in  = np.load('outputs/data/p_E_inside.npy')
# p_E_out = np.load('outputs/data/p_E_outside.npy')
# FPs_in  = np.load('outputs/data/FPs_inside.npy')
# FEs_in  = np.load('outputs/data/FEs_inside.npy')
# FPs_out = np.load('outputs/data/FPs_outside.npy')
# FEs_out = np.load('outputs/data/FEs_outside.npy')

# # same grid as solve_brt.py
# # GRID_RESOLUTION = (11, 11, 11, 11, 11, 11)
# # GRID_RESOLUTION = (21, 21, 21, 21, 21, 21)
# GRID_RESOLUTION = (15, 15, 15, 15, 15, 15) # medium try 2

# grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
#     # hj.sets.Box(
#     #     np.array([-5., -5., -5., -5., -5., -5.]),
#     #     np.array([ 5.,  5.,  5.,  5.,  5.,  5.])
#     # ),
#     hj.sets.Box(
#         np.array([-8., -8., -8., -8., -8., -8.]),
#         np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
#     ),
#     GRID_RESOLUTION
# )

# # interpolator for value function
# values_converged_interpolator = RegularGridInterpolator(
#     ([np.array(v) for v in grid.coordinate_vectors]),
#     np.array(values_converged),
#     bounds_error=False,
#     fill_value=None
# )

# # 2D slice of 6D value function
# # fix delta_py=0, delta_vx=0, delta_vy=0, delta_vz=0
# # plot delta_px vs delta_pz
# # like hw2: plot_value_and_safe_set_boundary

# # dpx = np.linspace(-5, 5, 101)
# # dpz = np.linspace(-5, 5, 101)
# dpx = np.linspace(-8, 8, 101)
# dpz = np.linspace(-8, 8, 101)

# DPX, DPZ = np.meshgrid(dpx, dpz)
# slice_pts = np.stack([
#     DPX.ravel(),
#     np.zeros_like(DPX.ravel()),  # delta_py = 0
#     DPZ.ravel(),
#     np.zeros_like(DPX.ravel()),  # delta_vx = 0
#     np.zeros_like(DPX.ravel()),  # delta_vy = 0
#     np.zeros_like(DPX.ravel()),  # delta_vz = 0
# ], axis=1)
# V_slice = values_converged_interpolator(slice_pts).reshape(DPX.shape)

# # hw2: fig, axes = plt.subplots(2,1)
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # plot 1: BRT slice + trajectories
# # like hw2: plot_value_and_safe_set_boundary + trajectory plot
# ax = axes[0]
# ax.pcolormesh(dpx, dpz, V_slice, cmap='RdBu', vmin=-3, vmax=3)
# ax.contour(dpx, dpz, V_slice, levels=[0], colors='k', linewidths=2)
# ax.plot(zs_in[:,  0], zs_in[:,  2], 'g', linewidth=2,
#         label=f'Inside BRT (z0={zs_in[0,:3]})')
# ax.plot(zs_out[:, 0], zs_out[:, 2], 'm', linewidth=2,
#         label=f'Outside BRT (z0={zs_out[0,:3]})')
# # capture radius circle
# theta_c = np.linspace(0, 2*np.pi, 100)
# ax.plot(np.cos(theta_c), np.sin(theta_c), 'r--', linewidth=1.5,
#         label='Capture radius')
# ax.set_xlabel(r'$\Delta p_x$ (m)')
# ax.set_ylabel(r'$\Delta p_z$ (m)')
# ax.set_title(r'BRT slice ($\Delta p_y=0$, $\Delta v=0$)')
# ax.legend(fontsize=7)
# ax.set_aspect('equal')

# # plot 2: relative distance over time
# # like hw2: control profile plot structure
# ax = axes[1]
# dt = 0.01
# t_in  = np.arange(len(zs_in))  * dt
# t_out = np.arange(len(zs_out)) * dt
# dist_in  = np.sqrt(zs_in[:,0]**2  + zs_in[:,1]**2  + zs_in[:,2]**2)
# dist_out = np.sqrt(zs_out[:,0]**2 + zs_out[:,1]**2 + zs_out[:,2]**2)
# ax.plot(t_in,  dist_in,  'g', linewidth=2, label='Inside BRT')
# ax.plot(t_out, dist_out, 'm', linewidth=2, label='Outside BRT')
# ax.axhline(1.0, color='r', linestyle='--', linewidth=1.5,
#            label='Capture radius')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel(r'$\|\Delta p\|$ (m)')
# ax.set_title('Relative distance over time')
# ax.legend()
# ax.grid(True)

# # plot 3: control profiles
# # like hw2: control profile plot
# ax = axes[2]
# ax.plot(t_in[:-1],  FPs_in,  'g',  linewidth=1.5, label='Pursuer (inside BRT)')
# ax.plot(t_in[:-1],  FEs_in,  'g--', linewidth=1.5, label='Evader (inside BRT)')
# ax.plot(t_out[:-1], FPs_out, 'm',  linewidth=1.5, label='Pursuer (outside BRT)')
# ax.plot(t_out[:-1], FEs_out, 'm--', linewidth=1.5, label='Evader (outside BRT)')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Thrust (N)')
# ax.set_title('Control profiles (bang-bang)')
# ax.legend(fontsize=7)
# ax.grid(True)

# fig.tight_layout()
# fig.savefig('outputs/plots/pursuit_evasion.png')
# print('Saved plot to outputs/plots/pursuit_evasion.png')

# dt = 0.01
# for zs, p_P, p_E, FPs, FEs, label in [
#     (zs_in,  p_P_in,  p_E_in,  FPs_in,  FEs_in,  'Inside BRT'),
#     (zs_out, p_P_out, p_E_out, FPs_out, FEs_out, 'Outside BRT')
# ]:
#     t      = np.arange(len(zs)) * dt
#     t_ctrl = np.arange(len(FPs)) * dt

#     fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

#     # plot 1: individual drone positions
#     ax = axes2[0]
#     ax.plot(t, p_P[:, 2], 'g', linewidth=2, label='Pursuer $p_z$')
#     ax.plot(t, p_E[:, 2], 'm', linewidth=2, label='Evader $p_z$')
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('$p_z$ (m)')
#     ax.set_title('Individual $p_z$ positions over time')
#     ax.legend()
#     ax.grid(True)

#     # plot 2: relative state over time
#     ax = axes2[1]
#     ax.plot(t, zs[:, 2], 'b', linewidth=2, label=r'$\Delta p_z$')
#     ax.plot(t, zs[:, 5], 'r', linewidth=2, label=r'$\Delta v_z$')
#     ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
#     ax.set_xlabel('Time (s)')
#     ax.set_title('Relative state over time')
#     ax.legend()
#     ax.grid(True)

#     # plot 3: control inputs
#     ax = axes2[2]
#     ax.plot(t_ctrl, FPs, 'g', linewidth=2, label='Pursuer $F_P$')
#     ax.plot(t_ctrl, FEs, 'm', linewidth=2, label='Evader $F_E$')
#     ax.axhline( F_P_MAX, color='g', linestyle=':', linewidth=0.8, label='$F_P$ max')
#     ax.axhline(-F_P_MAX, color='g', linestyle=':', linewidth=0.8)
#     ax.axhline( F_E_MAX, color='m', linestyle=':', linewidth=0.8, label='$F_E$ max')
#     ax.axhline(-F_E_MAX, color='m', linestyle=':', linewidth=0.8)
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('Thrust (N)')
#     ax.set_title('Control inputs (bang-bang)')
#     ax.legend()
#     ax.grid(True)

#     fig2.tight_layout()
#     save_name = label.lower().replace(' ', '_')
#     fig2.savefig(f'outputs/plots/trajectories_{save_name}.png')
#     print(f'Saved plot to outputs/plots/trajectories_{save_name}.png')


# def plot_brt_only(values_converged_interpolator):
#     """
#     Plot 2D slice of 6D BRT

#     Slice:
#         delta_py = 0
#         delta_vx = 0
#         delta_vy = 0
#         delta_vz = 0

#     Axes:
#         delta_px vs delta_pz
#     """
#     # dpx = np.linspace(-5, 5, 201)
#     # dpz = np.linspace(-5, 5, 201)
#     dpx = np.linspace(-8, 8, 201)
#     dpz = np.linspace(-8, 8, 201)

#     DPX, DPZ = np.meshgrid(dpx, dpz)

#     # 6D state slice
#     slice_pts = np.stack([
#         DPX.ravel(),                  # delta_px
#         np.zeros_like(DPX.ravel()),  # delta_py
#         DPZ.ravel(),                  # delta_pz
#         np.zeros_like(DPX.ravel()),  # delta_vx
#         np.zeros_like(DPX.ravel()),  # delta_vy
#         np.zeros_like(DPX.ravel()),  # delta_vz
#     ], axis=1)

#     # value func on slice
#     V_slice = values_converged_interpolator(slice_pts).reshape(DPX.shape)

#     fig, ax = plt.subplots(figsize=(7, 7))

#     # full v func heatmap
#     pcm = ax.pcolormesh(
#         dpx,
#         dpz,
#         V_slice,
#         cmap='RdBu',
#         shading='auto'
#     )

#     # BRT boundary: V = 0
#     contour = ax.contour(
#         dpx,
#         dpz,
#         V_slice,
#         levels=[0],
#         colors='k',
#         linewidths=2
#     )

#     # fill inside BRT (V < 0)
#     ax.contourf(
#         dpx,
#         dpz,
#         V_slice,
#         levels=[V_slice.min(), 0],
#         colors=['red'],
#         alpha=0.3
#     )

#     # capture radius circle
#     theta = np.linspace(0, 2*np.pi, 200)
#     ax.plot(
#         np.cos(theta),
#         np.sin(theta),
#         'g--',
#         linewidth=2,
#         label='Capture radius'
#     )

#     ax.set_xlabel(r'$\Delta p_x$ (m)')
#     ax.set_ylabel(r'$\Delta p_z$ (m)')
#     ax.set_title('Backward Reachable Tube (BRT)')
#     ax.set_aspect('equal')

#     plt.colorbar(pcm, ax=ax, label='Value Function V')
#     ax.legend()

#     fig.tight_layout()

#     fig.savefig('outputs/plots/brt_only.png')
#     print('Saved BRT-only plot to outputs/plots/brt_only.png')

# plot_brt_only(values_converged_interpolator)


# import numpy as np
# import jax.numpy as jnp
# import hj_reachability as hj
# from scipy.interpolate import RegularGridInterpolator
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')

# from dynamics import PursuitEvasion, F_P_MAX, F_E_MAX

# values_converged = np.load('outputs/data/values.npy')[-1]
# zs_in   = np.load('outputs/data/zs_inside.npy')
# zs_out  = np.load('outputs/data/zs_outside.npy')
# p_P_in  = np.load('outputs/data/p_P_inside.npy')
# p_P_out = np.load('outputs/data/p_P_outside.npy')
# p_E_in  = np.load('outputs/data/p_E_inside.npy')
# p_E_out = np.load('outputs/data/p_E_outside.npy')
# FPs_in  = np.load('outputs/data/FPs_inside.npy')
# FEs_in  = np.load('outputs/data/FEs_inside.npy')
# FPs_out = np.load('outputs/data/FPs_outside.npy')
# FEs_out = np.load('outputs/data/FEs_outside.npy')

# # same grid as solve_brt.py
# # GRID_RESOLUTION = (11, 11, 11, 11, 11, 11)
# # GRID_RESOLUTION = (21, 21, 21, 21, 21, 21)
# GRID_RESOLUTION = (15, 15, 15, 15, 15, 15) # medium try 2

# grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
#     # hj.sets.Box(
#     #     np.array([-5., -5., -5., -5., -5., -5.]),
#     #     np.array([ 5.,  5.,  5.,  5.,  5.,  5.])
#     # ),
#     hj.sets.Box(
#         np.array([-8., -8., -8., -8., -8., -8.]),
#         np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
#     ),
#     GRID_RESOLUTION
# )

# # interpolator for value function
# values_converged_interpolator = RegularGridInterpolator(
#     ([np.array(v) for v in grid.coordinate_vectors]),
#     np.array(values_converged),
#     bounds_error=False,
#     fill_value=None
# )

# # 2D slice of 6D value function
# # fix delta_py=0, delta_vx=0, delta_vy=0, delta_vz=0
# # plot delta_px vs delta_pz
# # like hw2: plot_value_and_safe_set_boundary

# # dpx = np.linspace(-5, 5, 101)
# # dpz = np.linspace(-5, 5, 101)
# dpx = np.linspace(-8, 8, 101)
# dpz = np.linspace(-8, 8, 101)

# DPX, DPZ = np.meshgrid(dpx, dpz)
# slice_pts = np.stack([
#     DPX.ravel(),
#     np.zeros_like(DPX.ravel()),  # delta_py = 0
#     DPZ.ravel(),
#     np.zeros_like(DPX.ravel()),  # delta_vx = 0
#     np.zeros_like(DPX.ravel()),  # delta_vy = 0
#     np.zeros_like(DPX.ravel()),  # delta_vz = 0
# ], axis=1)
# V_slice = values_converged_interpolator(slice_pts).reshape(DPX.shape)

# # hw2: fig, axes = plt.subplots(2,1)
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # plot 1: BRT slice + trajectories
# # like hw2: plot_value_and_safe_set_boundary + trajectory plot
# ax = axes[0]
# ax.pcolormesh(dpx, dpz, V_slice, cmap='RdBu', vmin=-3, vmax=3)
# ax.contour(dpx, dpz, V_slice, levels=[0], colors='k', linewidths=2)
# ax.plot(zs_in[:,  0], zs_in[:,  2], 'g', linewidth=2,
#         label=f'Inside BRT (z0={zs_in[0,:3]})')
# ax.plot(zs_out[:, 0], zs_out[:, 2], 'm', linewidth=2,
#         label=f'Outside BRT (z0={zs_out[0,:3]})')
# # capture radius circle
# theta_c = np.linspace(0, 2*np.pi, 100)
# ax.plot(np.cos(theta_c), np.sin(theta_c), 'r--', linewidth=1.5,
#         label='Capture radius')
# ax.set_xlabel(r'$\Delta p_x$ (m)')
# ax.set_ylabel(r'$\Delta p_z$ (m)')
# ax.set_title(r'BRT slice ($\Delta p_y=0$, $\Delta v=0$)')
# ax.legend(fontsize=7)
# ax.set_aspect('equal')

# # plot 2: relative distance over time
# # like hw2: control profile plot structure
# ax = axes[1]
# dt = 0.01
# t_in  = np.arange(len(zs_in))  * dt
# t_out = np.arange(len(zs_out)) * dt
# dist_in  = np.sqrt(zs_in[:,0]**2  + zs_in[:,1]**2  + zs_in[:,2]**2)
# dist_out = np.sqrt(zs_out[:,0]**2 + zs_out[:,1]**2 + zs_out[:,2]**2)
# ax.plot(t_in,  dist_in,  'g', linewidth=2, label='Inside BRT')
# ax.plot(t_out, dist_out, 'm', linewidth=2, label='Outside BRT')
# ax.axhline(1.0, color='r', linestyle='--', linewidth=1.5,
#            label='Capture radius')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel(r'$\|\Delta p\|$ (m)')
# ax.set_title('Relative distance over time')
# ax.legend()
# ax.grid(True)

# # plot 3: control profiles
# # like hw2: control profile plot
# ax = axes[2]
# ax.plot(t_in[:-1],  FPs_in,  'g',  linewidth=1.5, label='Pursuer (inside BRT)')
# ax.plot(t_in[:-1],  FEs_in,  'g--', linewidth=1.5, label='Evader (inside BRT)')
# ax.plot(t_out[:-1], FPs_out, 'm',  linewidth=1.5, label='Pursuer (outside BRT)')
# ax.plot(t_out[:-1], FEs_out, 'm--', linewidth=1.5, label='Evader (outside BRT)')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Thrust (N)')
# ax.set_title('Control profiles (bang-bang)')
# ax.legend(fontsize=7)
# ax.grid(True)

# fig.tight_layout()
# fig.savefig('outputs/plots/pursuit_evasion.png')
# print('Saved plot to outputs/plots/pursuit_evasion.png')

# dt = 0.01
# for zs, p_P, p_E, FPs, FEs, label in [
#     (zs_in,  p_P_in,  p_E_in,  FPs_in,  FEs_in,  'Inside BRT'),
#     (zs_out, p_P_out, p_E_out, FPs_out, FEs_out, 'Outside BRT')
# ]:
#     t      = np.arange(len(zs)) * dt
#     t_ctrl = np.arange(len(FPs)) * dt

#     fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

#     # plot 1: individual drone positions
#     ax = axes2[0]
#     ax.plot(t, p_P[:, 2], 'g', linewidth=2, label='Pursuer $p_z$')
#     ax.plot(t, p_E[:, 2], 'm', linewidth=2, label='Evader $p_z$')
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('$p_z$ (m)')
#     ax.set_title('Individual $p_z$ positions over time')
#     ax.legend()
#     ax.grid(True)

#     # plot 2: relative state over time
#     ax = axes2[1]
#     ax.plot(t, zs[:, 2], 'b', linewidth=2, label=r'$\Delta p_z$')
#     ax.plot(t, zs[:, 5], 'r', linewidth=2, label=r'$\Delta v_z$')
#     ax.axhline(0, color='k', linestyle=':', linewidth=0.8)
#     ax.set_xlabel('Time (s)')
#     ax.set_title('Relative state over time')
#     ax.legend()
#     ax.grid(True)

#     # plot 3: control inputs
#     ax = axes2[2]
#     ax.plot(t_ctrl, FPs, 'g', linewidth=2, label='Pursuer $F_P$')
#     ax.plot(t_ctrl, FEs, 'm', linewidth=2, label='Evader $F_E$')
#     ax.axhline( F_P_MAX, color='g', linestyle=':', linewidth=0.8, label='$F_P$ max')
#     ax.axhline(-F_P_MAX, color='g', linestyle=':', linewidth=0.8)
#     ax.axhline( F_E_MAX, color='m', linestyle=':', linewidth=0.8, label='$F_E$ max')
#     ax.axhline(-F_E_MAX, color='m', linestyle=':', linewidth=0.8)
#     ax.set_xlabel('Time (s)')
#     ax.set_ylabel('Thrust (N)')
#     ax.set_title('Control inputs (bang-bang)')
#     ax.legend()
#     ax.grid(True)

#     fig2.tight_layout()
#     save_name = label.lower().replace(' ', '_')
#     fig2.savefig(f'outputs/plots/trajectories_{save_name}.png')
#     print(f'Saved plot to outputs/plots/trajectories_{save_name}.png')


# def plot_brt_only(values_converged_interpolator):
#     """
#     Plot 2D slice of 6D BRT

#     Slice:
#         delta_py = 0
#         delta_vx = 0
#         delta_vy = 0
#         delta_vz = 0

#     Axes:
#         delta_px vs delta_pz
#     """
#     # dpx = np.linspace(-5, 5, 201)
#     # dpz = np.linspace(-5, 5, 201)
#     dpx = np.linspace(-8, 8, 201)
#     dpz = np.linspace(-8, 8, 201)

#     DPX, DPZ = np.meshgrid(dpx, dpz)

#     # 6D state slice
#     slice_pts = np.stack([
#         DPX.ravel(),                  # delta_px
#         np.zeros_like(DPX.ravel()),  # delta_py
#         DPZ.ravel(),                  # delta_pz
#         np.zeros_like(DPX.ravel()),  # delta_vx
#         np.zeros_like(DPX.ravel()),  # delta_vy
#         np.zeros_like(DPX.ravel()),  # delta_vz
#     ], axis=1)

#     # value func on slice
#     V_slice = values_converged_interpolator(slice_pts).reshape(DPX.shape)

#     fig, ax = plt.subplots(figsize=(7, 7))

#     # full v func heatmap
#     pcm = ax.pcolormesh(
#         dpx,
#         dpz,
#         V_slice,
#         cmap='RdBu',
#         shading='auto'
#     )

#     # BRT boundary: V = 0
#     contour = ax.contour(
#         dpx,
#         dpz,
#         V_slice,
#         levels=[0],
#         colors='k',
#         linewidths=2
#     )

#     # fill inside BRT (V < 0)
#     ax.contourf(
#         dpx,
#         dpz,
#         V_slice,
#         levels=[V_slice.min(), 0],
#         colors=['red'],
#         alpha=0.3
#     )

#     # capture radius circle
#     theta = np.linspace(0, 2*np.pi, 200)
#     ax.plot(
#         np.cos(theta),
#         np.sin(theta),
#         'g--',
#         linewidth=2,
#         label='Capture radius'
#     )

#     ax.set_xlabel(r'$\Delta p_x$ (m)')
#     ax.set_ylabel(r'$\Delta p_z$ (m)')
#     ax.set_title('Backward Reachable Tube (BRT)')
#     ax.set_aspect('equal')

#     plt.colorbar(pcm, ax=ax, label='Value Function V')
#     ax.legend()

#     fig.tight_layout()

#     fig.savefig('outputs/plots/brt_only.png')
#     print('Saved BRT-only plot to outputs/plots/brt_only.png')

# plot_brt_only(values_converged_interpolator)

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
    values_all: shape (n_times, 15, 15, 15, 15, 15, 15)
    times: shape (n_times,) -- negative, from 0 to -10
    """
    # pick a few representative timesteps to show
    # times goes from 0 to -10, we want early/mid/late/converged
    n_times = len(times)
    # indices: t=0 (initial=capture set), a few intermediate, t=-10 (converged)
    indices = [0, n_times//10, n_times//8, n_times//6, n_times//4, n_times//3, n_times//2, n_times-1]
    selected_times = [times[i] for i in indices]

    dpx = np.linspace(-8, 8, 201)
    dpz = np.linspace(-8, 8, 201)
    dvz = np.linspace(-8, 8, 201)
    # dpx = np.linspace(-10, 10, 201)
    # dpz = np.linspace(-10, 10, 201)
    # dvz = np.linspace(-10, 10, 201)

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

    fig, axes = plt.subplots(2, len(indices), figsize=(5*len(indices), 10))

    for col, (idx, t) in enumerate(zip(indices, selected_times)):
        # build interpolator for this timestep
        interp_t = RegularGridInterpolator(
            ([np.array(v) for v in grid.coordinate_vectors]),
            np.array(values_all[idx]),
            bounds_error=False, fill_value=None
        )

        V_pos = interp_t(slice_pos_pts).reshape(DPX.shape)
        V_vel = interp_t(slice_vel_pts).reshape(DPZ_vel.shape)

        # row 0: px vs pz
        ax = axes[0, col]
        ax.pcolormesh(dpx, dpz, V_pos, cmap='RdBu', shading='auto')
        ax.contour(dpx, dpz, V_pos, levels=[0], colors='k', linewidths=2)
        ax.contourf(dpx, dpz, V_pos, levels=[V_pos.min(), 0], colors=['red'], alpha=0.3)
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'g--', linewidth=1.5)
        ax.set_title(f't = {t:.1f}s', fontsize=11)
        ax.set_xlabel(r'$\Delta p_x$ (m)')
        if col == 0:
            ax.set_ylabel(r'$\Delta p_z$ (m)')
        ax.set_aspect('equal')

        # row 1: pz vs vz
        ax = axes[1, col]
        ax.pcolormesh(dpz, dvz, V_vel, cmap='RdBu', shading='auto')
        ax.contour(dpz, dvz, V_vel, levels=[0], colors='k', linewidths=2)
        ax.contourf(dpz, dvz, V_vel, levels=[V_vel.min(), 0], colors=['red'], alpha=0.3)
        ax.axvline(-1.0, color='g', linestyle='--', linewidth=1.5)
        ax.axvline( 1.0, color='g', linestyle='--', linewidth=1.5)
        ax.set_xlabel(r'$\Delta p_z$ (m)')
        if col == 0:
            ax.set_ylabel(r'$\Delta v_z$ (m/s)')

    axes[0, 0].set_title(f't = {selected_times[0]:.1f}s (initial)', fontsize=11)
    axes[0, -1].set_title(f't = {selected_times[-1]:.1f}s (converged)', fontsize=11)

    fig.suptitle('BRT Evolution Over Time', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig('outputs/plots/brt_over_time.png', bbox_inches='tight')
    print('Saved outputs/plots/brt_over_time.png')


plot_brt_only(values_converged_interpolator)
values_all = np.load('outputs/data/values.npy')
times_all  = np.load('outputs/data/times.npy')
plot_brt_over_time(values_all, times_all, values_converged_interpolator)