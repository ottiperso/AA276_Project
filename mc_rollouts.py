"""
MC Rollouts + Controller Comparison Script
------------------------------------------
1. MC rollouts: runs many perturbed trajectories from a few base ICs,
   plots them on the pz-vz phase plane to show BRT as separating surface.

2. Controller comparison: compares optimal HJI bang-bang controller vs
   naive controller (always thrust +z) across all ICs, saves to JSON.

Usage:
  python mc_rollouts.py
"""

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from dynamics import PursuitEvasion, F_P_MAX, F_E_MAX

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/data', exist_ok=True)

# ── load BRT ──
values = np.load('outputs/data/values.npy')
values_converged = values[-1]

GRID_RESOLUTION = (15, 15, 15, 15, 15, 15)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-8., -8., -8., -8., -8., -8.]),
        np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
    ),
    GRID_RESOLUTION
)
solver_settings = hj.SolverSettings.with_accuracy(
    'very_high',
    hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
)

GRID_LO = np.array([-8., -8., -8., -8., -8., -8.])
GRID_HI = np.array([ 8.,  8.,  8.,  8.,  8.,  8.])

values_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(values_converged),
    bounds_error=False, fill_value=None
)

grads_converged = grid.grad_values(
    jnp.array(values_converged),
    solver_settings.upwind_scheme
)
beta5s_converged = grads_converged[..., 5]
beta5s_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(beta5s_converged),
    bounds_error=False, fill_value=None
)

# ── controllers ──
def optimal_pursuer(z):
    z_clipped = np.clip(z, GRID_LO + 0.1, GRID_HI - 0.1)
    grad = beta5s_converged_interpolator(z_clipped.reshape(1, -1)).item()
    if abs(grad) < 1e-6:
        return -np.sign(z[2]) * F_P_MAX if abs(z[2]) > 0.01 else F_P_MAX
    return -np.sign(grad) * F_P_MAX

def naive_pursuer(z):
    # always thrust in +z (toward evader if evader is above)
    # simple heuristic: thrust toward evader based on sign of delta_pz
    return -np.sign(z[2]) * F_P_MAX if abs(z[2]) > 0.01 else F_P_MAX

def optimal_evader(z):
    z_clipped = np.clip(z, GRID_LO + 0.1, GRID_HI - 0.1)
    grad = beta5s_converged_interpolator(z_clipped.reshape(1, -1)).item()
    return -np.sign(grad) * F_E_MAX

def euler_step(z, F_P, F_E, dt=0.01):
    dzdt = np.array([z[3], z[4], z[5], 0., 0., float(F_P) - float(F_E)])
    return z + dt * dzdt

def simulate_simple(z0, pursuer_fn, evader_fn, nt, dt=0.01):
    """Lightweight simulate — returns zs and outcome only."""
    z = z0.copy()
    zs = [z.copy()]
    captured = False
    capture_t = None

    for step in range(nt):
        if np.any(z < GRID_LO) or np.any(z > GRID_HI):
            break
        F_P = pursuer_fn(z)
        F_E = evader_fn(z)
        z = euler_step(z, F_P, F_E, dt)
        zs.append(z.copy())
        dist = np.sqrt(z[0]**2 + z[1]**2 + z[2]**2)
        if dist <= 1.0:
            captured = True
            capture_t = (step + 1) * dt
            break

    return np.array(zs), captured, capture_t


dt = 0.01
nt = int(8.0 / dt)
np.random.seed(0)

# ── 1. MC ROLLOUTS ──
print('Running MC rollouts...')

# 3 base points: inside, boundary, outside
mc_bases = {
    'inside':   np.array([0., 0., -1.5, 0., 0.,  1.5]),
    'boundary': np.array([0., 0.,  1.545, 0., 0., 0.773]),
    'outside':  np.array([0., 0.,  4.0, 0., 0.,  2.0]),
}
n_rollouts   = 50
noise_scales = {'inside': 0.3, 'boundary': 0.15, 'outside': 0.3}
mc_colors    = {'inside': 'green', 'boundary': 'orange', 'outside': 'red'}

# precompute BRT slice for background
dpz = np.linspace(-8, 8, 100)
dvz = np.linspace(-8, 8, 100)
DPZ, DVZ = np.meshgrid(dpz, dvz)
pts = np.stack([
    np.zeros_like(DPZ.ravel()), np.zeros_like(DPZ.ravel()), DPZ.ravel(),
    np.zeros_like(DPZ.ravel()), np.zeros_like(DPZ.ravel()), DVZ.ravel(),
], axis=1)
V_bg = values_converged_interpolator(pts).reshape(DPZ.shape)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for col, (base_name, z0_base) in enumerate(mc_bases.items()):
    ax = axes[col]
    ax.pcolormesh(dpz, dvz, V_bg, cmap='RdBu', shading='auto',
                  vmin=-3, vmax=3, alpha=0.7)
    ax.contour(dpz, dvz, V_bg, levels=[0], colors='k', linewidths=2)
    ax.contourf(dpz, dvz, V_bg, levels=[V_bg.min(), 0],
                colors=['red'], alpha=0.15)
    ax.axvline(-1.0, color='g', linestyle='--', linewidth=1.5)
    ax.axvline( 1.0, color='g', linestyle='--', linewidth=1.5)

    color = mc_colors[base_name]
    n_captured = 0

    for i in range(n_rollouts):
        noise = np.random.randn(6) * noise_scales[base_name]
        noise[0:2] = 0   # keep px,py = 0
        noise[3:5] = 0   # keep vx,vy = 0
        z0 = z0_base + noise

        # clip to grid
        z0 = np.clip(z0, GRID_LO + 0.1, GRID_HI - 0.1)

        zs, captured, _ = simulate_simple(z0, optimal_pursuer, optimal_evader, nt, dt)
        if captured:
            n_captured += 1

        alpha = 0.5 if captured else 0.3
        lw    = 1.0 if captured else 0.8
        ls    = '-' if captured else '--'
        ax.plot(zs[:, 2], zs[:, 5], color=color, alpha=alpha,
                linewidth=lw, linestyle=ls)
        ax.plot(zs[0, 2], zs[0, 5], 'o', color=color,
                markersize=4, alpha=0.7)

    # mark base point
    V0 = values_converged_interpolator(z0_base.reshape(1,-1)).item()
    ax.plot(z0_base[2], z0_base[5], '*', color='k', markersize=12,
            zorder=5, label=f'Base IC ($V={V0:.2f}$)')

    ax.set_xlabel(r'$\Delta p_z$ (m)', fontsize=12)
    ax.set_ylabel(r'$\Delta v_z$ (m/s)', fontsize=12)
    ax.set_title(f'MC rollouts: {base_name}\n'
                 f'({n_captured}/{n_rollouts} captured, '
                 f'solid=captured, dashed=escaped)',
                 fontsize=10)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Monte Carlo Rollouts from Three Initial Regions\n'
             '(50 perturbed trajectories each, optimal controllers)',
             fontsize=13)
fig.tight_layout()
fig.savefig('outputs/plots/mc_rollouts.png', bbox_inches='tight')
print('Saved outputs/plots/mc_rollouts.png')


# ── 2. CONTROLLER COMPARISON ──
print('\nRunning controller comparison...')

all_ics = {
    'inside_brt':   np.array([0., 0., -1.5,  0., 0.,  1.5]),
    'inside_far':   np.array([0., 0.,  1.5,  0., 0., -0.5]),
    'boundary':     np.array([0., 0.,  2.0,  0., 0.,  0.5]),
    'outside_near': np.array([0., 0.,  3.5,  0., 0.,  1.75]),
    'outside_far':  np.array([0., 0.,  4.0,  0., 0.,  2.0]),
}

comparison_results = {}

for ic_name, z0 in all_ics.items():
    V0 = values_converged_interpolator(z0.reshape(1,-1)).item()
    brt_class = 'inside' if V0 < 0 else ('boundary' if abs(V0) < 0.1 else 'outside')

    # optimal pursuer vs optimal evader
    _, opt_captured, opt_t = simulate_simple(
        z0, optimal_pursuer, optimal_evader, nt, dt)

    # naive pursuer vs optimal evader
    _, naive_captured, naive_t = simulate_simple(
        z0, naive_pursuer, optimal_evader, nt, dt)

    comparison_results[ic_name] = {
        'z0':          z0.tolist(),
        'V0':          round(V0, 4),
        'brt_class':   brt_class,
        'optimal': {
            'captured':   opt_captured,
            'capture_t':  round(opt_t, 3) if opt_t else None,
        },
        'naive': {
            'captured':   naive_captured,
            'capture_t':  round(naive_t, 3) if naive_t else None,
        },
    }

    print(f'{ic_name:15s}  V={V0:+.4f}  '
          f'optimal={"captured @ "+f"{opt_t:.2f}s" if opt_captured else "escaped":20s}  '
          f'naive={"captured @ "+f"{naive_t:.2f}s" if naive_captured else "escaped"}')

with open('outputs/data/controller_comparison.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)
print('\nSaved outputs/data/controller_comparison.json')


# ── 3. CONTROLLER COMPARISON PLOT ──
fig2, axes2 = plt.subplots(2, len(all_ics), figsize=(5*len(all_ics), 10))

for col, (ic_name, z0) in enumerate(all_ics.items()):
    V0 = comparison_results[ic_name]['V0']

    for row, (pursuer_fn, label, color) in enumerate([
        (optimal_pursuer, 'Optimal', 'blue'),
        (naive_pursuer,   'Naive',   'orange'),
    ]):
        ax = axes2[row, col]
        ax.pcolormesh(dpz, dvz, V_bg, cmap='RdBu', shading='auto',
                      vmin=-3, vmax=3, alpha=0.6)
        ax.contour(dpz, dvz, V_bg, levels=[0], colors='k', linewidths=1.5)
        ax.axvline(-1.0, color='g', linestyle='--', linewidth=1.2)
        ax.axvline( 1.0, color='g', linestyle='--', linewidth=1.2)

        zs, captured, cap_t = simulate_simple(z0, pursuer_fn, optimal_evader, nt, dt)
        ax.plot(zs[:, 2], zs[:, 5], color=color, linewidth=2)
        ax.plot(zs[0, 2], zs[0, 5], 'o', color=color, markersize=8,
                markeredgecolor='k')

        outcome = f'Captured @ {cap_t:.2f}s' if captured else 'Escaped'
        ax.set_title(f'{ic_name}\n{label}: {outcome}', fontsize=9)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_xlabel(r'$\Delta p_z$ (m)', fontsize=9)
        if col == 0:
            ax.set_ylabel(f'{label}\n' + r'$\Delta v_z$ (m/s)', fontsize=9)
        ax.grid(True, alpha=0.3)

fig2.suptitle('Optimal vs Naive Pursuer Controller Comparison\n'
              '(top: optimal HJI bang-bang, bottom: naive sign-based)',
              fontsize=13)
fig2.tight_layout()
fig2.savefig('outputs/plots/controller_comparison.png', bbox_inches='tight')
print('Saved outputs/plots/controller_comparison.png')

print('\nDone! Files saved:')
print('  outputs/plots/mc_rollouts.png')
print('  outputs/plots/controller_comparison.png')
print('  outputs/data/controller_comparison.json')