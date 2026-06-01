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
matplotlib.use('Agg')

from dynamics import PursuitEvasion, F_P_MAX, F_E_MAX

values_converged = np.load('outputs/data/values.npy')[-1]
zs_in   = np.load('outputs/data/zs_inside.npy')
zs_out  = np.load('outputs/data/zs_outside.npy')
p_P_in  = np.load('outputs/data/p_P_inside.npy')
p_P_out = np.load('outputs/data/p_P_outside.npy')
p_E_in  = np.load('outputs/data/p_E_inside.npy')
p_E_out = np.load('outputs/data/p_E_outside.npy')
FPs_in  = np.load('outputs/data/FPs_inside.npy')
FEs_in  = np.load('outputs/data/FEs_inside.npy')
FPs_out = np.load('outputs/data/FPs_outside.npy')
FEs_out = np.load('outputs/data/FEs_outside.npy')

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

# 2D slice: delta_pz (x-axis) vs delta_vz (y-axis)
# meshgrid(dpz, dvz): DPZ[i,j]=dpz[j] (x, varies cols), DVZ[i,j]=dvz[i] (y, varies rows)
# pcolormesh(dpz, dvz, V) expects V[i,j] = value at (dpz[j], dvz[i]) -- matches reshape(DVZ.shape)
dpz = np.linspace(-8, 8, 101)
dvz = np.linspace(-8, 8, 101)

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

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# plot 1: BRT slice + trajectories
ax = axes[0]
ax.pcolormesh(dpz, dvz, V_slice, cmap='RdBu', vmin=-3, vmax=3)
ax.contour(dpz, dvz, V_slice, levels=[0], colors='k', linewidths=2)
ax.plot(zs_in[:,  2], zs_in[:,  5], 'g', linewidth=2,
        label=f'Inside BRT (z0={zs_in[0,:3]})')
ax.plot(zs_out[:, 2], zs_out[:, 5], 'm', linewidth=2,
        label=f'Outside BRT (z0={zs_out[0,:3]})')
ax.axvline(-1.0, color='r', linestyle='--', linewidth=1.5, label='Capture radius')
ax.axvline( 1.0, color='r', linestyle='--', linewidth=1.5)
ax.set_xlabel(r'$\Delta p_z$ (m)')
ax.set_ylabel(r'$\Delta v_z$ (m/s)')
ax.set_title(r'BRT slice ($\Delta p_x=\Delta p_y=0$, $\Delta v_x=\Delta v_y=0$)')
ax.legend(fontsize=7)
# no set_aspect('equal') -- dpz and dvz have different units

# plot 2: relative distance over time
ax = axes[1]
dt = 0.01
t_in  = np.arange(len(zs_in))  * dt
t_out = np.arange(len(zs_out)) * dt
dist_in  = np.sqrt(zs_in[:,0]**2  + zs_in[:,1]**2  + zs_in[:,2]**2)
dist_out = np.sqrt(zs_out[:,0]**2 + zs_out[:,1]**2 + zs_out[:,2]**2)
ax.plot(t_in,  dist_in,  'g', linewidth=2, label='Inside BRT')
ax.plot(t_out, dist_out, 'm', linewidth=2, label='Outside BRT')
ax.axhline(1.0, color='r', linestyle='--', linewidth=1.5, label='Capture radius')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\|\Delta p\|$ (m)')
ax.set_title('Relative distance over time')
ax.legend()
ax.grid(True)

# plot 3: control profiles
ax = axes[2]
ax.plot(t_in[:-1],  FPs_in,  'g',   linewidth=1.5, label='Pursuer (inside BRT)')
ax.plot(t_in[:-1],  FEs_in,  'g--', linewidth=1.5, label='Evader (inside BRT)')
ax.plot(t_out[:-1], FPs_out, 'm',   linewidth=1.5, label='Pursuer (outside BRT)')
ax.plot(t_out[:-1], FEs_out, 'm--', linewidth=1.5, label='Evader (outside BRT)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Thrust (N)')
ax.set_title('Control profiles (bang-bang)')
ax.legend(fontsize=7)
ax.grid(True)

fig.tight_layout()
fig.savefig('outputs/plots/pursuit_evasion.png')
print('Saved plot to outputs/plots/pursuit_evasion.png')

dt = 0.01
for zs, p_P, p_E, FPs, FEs, label in [
    (zs_in,  p_P_in,  p_E_in,  FPs_in,  FEs_in,  'Inside BRT'),
    (zs_out, p_P_out, p_E_out, FPs_out, FEs_out, 'Outside BRT')
]:
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
    save_name = label.lower().replace(' ', '_')
    fig2.savefig(f'outputs/plots/trajectories_{save_name}.png')
    print(f'Saved plot to outputs/plots/trajectories_{save_name}.png')


def plot_brt_only(values_converged_interpolator):
    """
    Two-panel BRT plot:
      Left:  delta_px (x) vs delta_pz (y), slice at delta_py=0, delta_v=0
      Right: delta_pz (x) vs delta_vz (y), slice at delta_px=delta_py=0, delta_vx=delta_vy=0
    """
    dpx = np.linspace(-8, 8, 201)
    dpz = np.linspace(-8, 8, 201)
    dvz = np.linspace(-8, 8, 201)

    # Panel 1: DPX varies cols (x), DPZ_pos varies rows (y)
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

    # Panel 2: DPZ_vel varies cols (x), DVZ varies rows (y)
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

    # Panel 1: delta_px vs delta_pz
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

    # Panel 2: delta_pz vs delta_vz
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
    # no set_aspect('equal') -- different units
    plt.colorbar(pcm2, ax=ax, label='Value Function V')
    ax.legend()

    fig.tight_layout()
    fig.savefig('outputs/plots/brt_only.png')
    print('Saved BRT-only plot to outputs/plots/brt_only.png')

plot_brt_only(values_converged_interpolator)