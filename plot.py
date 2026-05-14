import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from dynamics import PursuitEvasion

# load outputs
values_converged = np.load('outputs/values.npy')[-1]
zs_in   = np.load('outputs/zs_inside.npy')
zs_out  = np.load('outputs/zs_outside.npy')
FPs_in  = np.load('outputs/FPs_inside.npy')
FEs_in  = np.load('outputs/FEs_inside.npy')
FPs_out = np.load('outputs/FPs_outside.npy')
FEs_out = np.load('outputs/FEs_outside.npy')

# same grid as solve_brt.py
GRID_RESOLUTION = (11, 11, 11, 11, 11, 11)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-5., -5., -5., -5., -5., -5.]),
        np.array([ 5.,  5.,  5.,  5.,  5.,  5.])
    ),
    GRID_RESOLUTION
)

# interpolator for value function
values_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(values_converged),
    bounds_error=False,
    fill_value=None
)

# 2D slice of 6D value function
# fix delta_py=0, delta_vx=0, delta_vy=0, delta_vz=0
# plot delta_px vs delta_pz
# like hw2: plot_value_and_safe_set_boundary
dpx = np.linspace(-5, 5, 101)
dpz = np.linspace(-5, 5, 101)
DPX, DPZ = np.meshgrid(dpx, dpz)
slice_pts = np.stack([
    DPX.ravel(),
    np.zeros_like(DPX.ravel()),  # delta_py = 0
    DPZ.ravel(),
    np.zeros_like(DPX.ravel()),  # delta_vx = 0
    np.zeros_like(DPX.ravel()),  # delta_vy = 0
    np.zeros_like(DPX.ravel()),  # delta_vz = 0
], axis=1)
V_slice = values_converged_interpolator(slice_pts).reshape(DPX.shape)

# hw2: fig, axes = plt.subplots(2,1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# plot 1: BRT slice + trajectories
# like hw2: plot_value_and_safe_set_boundary + trajectory plot
ax = axes[0]
ax.pcolormesh(dpx, dpz, V_slice, cmap='RdBu', vmin=-3, vmax=3)
ax.contour(dpx, dpz, V_slice, levels=[0], colors='k', linewidths=2)
ax.plot(zs_in[:,  0], zs_in[:,  2], 'g', linewidth=2,
        label=f'Inside BRT (z0={zs_in[0,:3]})')
ax.plot(zs_out[:, 0], zs_out[:, 2], 'm', linewidth=2,
        label=f'Outside BRT (z0={zs_out[0,:3]})')
# capture radius circle
theta_c = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta_c), np.sin(theta_c), 'r--', linewidth=1.5,
        label='Capture radius')
ax.set_xlabel(r'$\Delta p_x$ (m)')
ax.set_ylabel(r'$\Delta p_z$ (m)')
ax.set_title(r'BRT slice ($\Delta p_y=0$, $\Delta v=0$)')
ax.legend(fontsize=7)
ax.set_aspect('equal')

# plot 2: relative distance over time
# like hw2: control profile plot structure
ax = axes[1]
dt = 0.01
t_in  = np.arange(len(zs_in))  * dt
t_out = np.arange(len(zs_out)) * dt
dist_in  = np.sqrt(zs_in[:,0]**2  + zs_in[:,1]**2  + zs_in[:,2]**2)
dist_out = np.sqrt(zs_out[:,0]**2 + zs_out[:,1]**2 + zs_out[:,2]**2)
ax.plot(t_in,  dist_in,  'g', linewidth=2, label='Inside BRT')
ax.plot(t_out, dist_out, 'm', linewidth=2, label='Outside BRT')
ax.axhline(1.0, color='r', linestyle='--', linewidth=1.5,
           label='Capture radius')
ax.set_xlabel('Time (s)')
ax.set_ylabel(r'$\|\Delta p\|$ (m)')
ax.set_title('Relative distance over time')
ax.legend()
ax.grid(True)

# plot 3: control profiles
# like hw2: control profile plot
ax = axes[2]
ax.plot(t_in[:-1],  FPs_in,  'g',  linewidth=1.5, label='Pursuer (inside BRT)')
ax.plot(t_in[:-1],  FEs_in,  'g--', linewidth=1.5, label='Evader (inside BRT)')
ax.plot(t_out[:-1], FPs_out, 'm',  linewidth=1.5, label='Pursuer (outside BRT)')
ax.plot(t_out[:-1], FEs_out, 'm--', linewidth=1.5, label='Evader (outside BRT)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Thrust (N)')
ax.set_title('Control profiles (bang-bang)')
ax.legend(fontsize=7)
ax.grid(True)

fig.tight_layout()
fig.savefig('outputs/pursuit_evasion.png')
print('Saved plot to outputs/pursuit_evasion.png')