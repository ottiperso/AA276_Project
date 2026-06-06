# validate_brt.py
# Monte Carlo validation of the BRT guarantee

"""
Samples random states from state space, propagates them forward
under optimal (bang-bang) control for pursuer and evader, and
gives stats on BRT guarantee violations:
  - Trajectories that started INSIDE BRT but never reached failure set (violations)
  - Trajectories that started OUTSIDE BRT but entered failure set
  - Trajectories that exited the grid domain
  - Summary
"""

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import argparse
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from dynamics import PursuitEvasion, F_P_MAX, F_E_MAX

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/data', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=2000, help='num of random initial states')
parser.add_argument('--t_horizon', type=float, default=15.0, help='forward sim time (s)')
parser.add_argument('--dt', type=float, default=0.01, help='Euler step size (s)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--sample_inside_only', action='store_true', help='sample from near BRT for better BRT coverage')
args = parser.parse_args()

np.random.seed(args.seed)

# load BRT
values = np.load('outputs/data/values.npy')
times = np.load('outputs/data/times.npy')
values_converged = values[-1]

GRID_RESOLUTION = (15, 15, 15, 15, 15, 15)
R_CAPTURE = 1.0

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-8., -8., -8., -8., -8., -8.]),
        np.array([ 8., 8., 8., 8., 8., 8.])
    ),
    GRID_RESOLUTION
)
solver_settings = hj.SolverSettings.with_accuracy('very_high',
    hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
)

GRID_LO = np.array([-8., -8., -8., -8., -8., -8.])
GRID_HI = np.array([ 8., 8., 8., 8., 8., 8.])

values_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(values_converged),
    bounds_error=False,
    fill_value=None
)
grads_converged = grid.grad_values(
    jnp.array(values_converged),
    solver_settings.upwind_scheme
)
beta5s_converged = grads_converged[..., 5]
beta5s_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(beta5s_converged),
    bounds_error=False,
    fill_value=None
)

# controllers, same as in simulate.py
def pursuer_control(z):
    z_clipped = np.clip(z, GRID_LO + 0.1, GRID_HI - 0.1)
    grad = beta5s_converged_interpolator(z_clipped.reshape(1, -1)).item()
    if abs(grad) < 1e-6:
        return -np.sign(z[2]) * F_P_MAX if abs(z[2]) > 0.01 else F_P_MAX
    return -np.sign(grad) * F_P_MAX

def evader_control(z):
    z_clipped = np.clip(z, GRID_LO + 0.1, GRID_HI - 0.1)
    grad = beta5s_converged_interpolator(z_clipped.reshape(1, -1)).item()
    return -np.sign(grad) * F_E_MAX

def euler_step(z, F_P, F_E, dt):
    dzdt = np.array([z[3], z[4], z[5], 0., 0., float(F_P) - float(F_E)])
    return z + dt * dzdt

def in_failure_set(z):
    return np.sqrt(z[0]**2 + z[1]**2 + z[2]**2) <= R_CAPTURE

def outside_grid(z):
    return np.any(z < GRID_LO) or np.any(z > GRID_HI)

# some sample initial states
if args.sample_inside_only:
    # sample from a tighter box around the BRT
    # BRT shape: pz in ~[-3,3], vz in ~[-3,3], px/py/vx/vy small
    sample_lo = np.array([-2., -2., -3., -2., -2., -3.])
    sample_hi = np.array([ 2., 2., 3., 2., 2., 3.])
    print(f'Sampling from near BRT: {sample_lo} to {sample_hi}')
else:
    sample_lo = GRID_LO
    sample_hi = GRID_HI
    print('Sampling uniformly from full grid')

z0_samples = np.random.uniform(
    low=sample_lo, high=sample_hi,
    size=(args.n_samples, 6)
)

# V at each sample
V0s = values_converged_interpolator(z0_samples).ravel()
inside_brt_mask = V0s < 0
outside_brt_mask = V0s >= 0

print(f'Total samples: {args.n_samples}')
print(f'Inside BRT (V<0): {inside_brt_mask.sum()}')
print(f'Outside BRT (V>=0): {outside_brt_mask.sum()}')
print(f'Time horizon: {args.t_horizon}s')
print(f'dt: {args.dt}s')
print(f'Steps per trajectory: {int(args.t_horizon / args.dt)}')

nt = int(args.t_horizon / args.dt)

# results
outcomes = []

failing_trajectories = [] # inside BRT, never captured (BRT violations)
surprise_captures = [] # outside BRT, got captured anyway
grid_exits = [] # left grid domain

for idx in tqdm(range(args.n_samples), desc='Simulating'):
    z0 = z0_samples[idx]
    V0 = float(V0s[idx])
    started_inside = bool(inside_brt_mask[idx])

    z = z0.copy()
    trajectory = [z.copy()]
    captured = False
    exited_grid = False
    capture_t = None
    exit_t = None

    for step in range(nt):
        if outside_grid(z):
            exited_grid = True
            exit_t = step * args.dt
            break

        F_P = pursuer_control(z)
        F_E = evader_control(z)
        z = euler_step(z, F_P, F_E, args.dt)
        trajectory.append(z.copy())

        if in_failure_set(z):
            captured = True
            capture_t = (step + 1) * args.dt
            break

    outcome = dict(
        idx = idx,
        z0 = z0.tolist(),
        V0 = V0,
        started_inside = started_inside,
        captured = captured,
        capture_t = capture_t,
        exited_grid = exited_grid,
        exit_t = exit_t,
        n_steps = len(trajectory),
    )
    outcomes.append(outcome)

    traj_arr = np.array(trajectory)

    if started_inside and not captured and not exited_grid:
        failing_trajectories.append({
            'idx': idx,
            'z0': z0.tolist(),
            'V0': V0,
            'trajectory': traj_arr.tolist(),
        })

    if not started_inside and captured:
        surprise_captures.append({
            'idx': idx,
            'z0': z0.tolist(),
            'V0': V0,
            'capture_t': capture_t,
        })

    if exited_grid:
        grid_exits.append({
            'idx': idx,
            'z0': z0.tolist(),
            'V0': V0,
            'started_inside': started_inside,
            'exit_t': exit_t,
        })

# stats
inside_indices = [i for i, o in enumerate(outcomes) if o['started_inside']]
outside_indices = [i for i, o in enumerate(outcomes) if not o['started_inside']]

inside_captured = [i for i in inside_indices if outcomes[i]['captured']]
inside_not_captured = [i for i in inside_indices if not outcomes[i]['captured'] and not outcomes[i]['exited_grid']]
inside_exited = [i for i in inside_indices if outcomes[i]['exited_grid']]
outside_captured = [i for i in outside_indices if outcomes[i]['captured']]
outside_not_cap = [i for i in outside_indices if not outcomes[i]['captured'] and not outcomes[i]['exited_grid']]
outside_exited = [i for i in outside_indices if outcomes[i]['exited_grid']]

capture_times_inside = [outcomes[i]['capture_t'] for i in inside_captured]

print('\n' + '='*55)
print('BRT VALIDATION RESULTS')
print('='*55)
print(f'\nINSIDE BRT ({len(inside_indices)} samples):')
print(f' Captured (correct): {len(inside_captured):4d} ({100*len(inside_captured)/max(1,len(inside_indices)):.1f}%)')
print(f' Not captured (VIOLATIONS): {len(inside_not_captured):4d} ({100*len(inside_not_captured)/max(1,len(inside_indices)):.1f}%)')
print(f' Exited grid: {len(inside_exited):4d} ({100*len(inside_exited)/max(1,len(inside_indices)):.1f}%)')
if capture_times_inside:
    print(f' Mean capture time: {np.mean(capture_times_inside):.2f}s')
    print(f' Median capture time: {np.median(capture_times_inside):.2f}s')
    print(f' Max capture time: {np.max(capture_times_inside):.2f}s')

print(f'\nOUTSIDE BRT ({len(outside_indices)} samples):')
print(f' Not captured (correct): {len(outside_not_cap):4d} ({100*len(outside_not_cap)/max(1,len(outside_indices)):.1f}%)')
print(f' Captured (notable): {len(outside_captured):4d} ({100*len(outside_captured)/max(1,len(outside_indices)):.1f}%)')
print(f' Exited grid: {len(outside_exited):4d} ({100*len(outside_exited)/max(1,len(outside_indices)):.1f}%)')

print(f'\nBRT GUARANTEE VIOLATION RATE: {len(inside_not_captured)}/{len(inside_indices)} = {100*len(inside_not_captured)/max(1,len(inside_indices)):.2f}%')
print('(violations expected ~0% for correct BRT; nonzero due to grid resolution)')
print('='*55)

suffix = '_targeted' if args.sample_inside_only else '_uniform'

summary = dict(
    n_samples = args.n_samples,
    t_horizon = args.t_horizon,
    dt = args.dt,
    seed = args.seed,
    sample_mode = 'targeted' if args.sample_inside_only else 'uniform',
    n_inside_brt = len(inside_indices),
    n_outside_brt = len(outside_indices),
    n_inside_captured = len(inside_captured),
    n_inside_violations = len(inside_not_captured),
    n_inside_exited = len(inside_exited),
    n_outside_captured = len(outside_captured),
    n_outside_not_cap = len(outside_not_cap),
    n_outside_exited = len(outside_exited),
    capture_time_mean = float(np.mean(capture_times_inside)) if capture_times_inside else None,
    capture_time_median = float(np.median(capture_times_inside)) if capture_times_inside else None,
    capture_time_max = float(np.max(capture_times_inside)) if capture_times_inside else None,
    violation_rate = len(inside_not_captured) / max(1, len(inside_indices)),
)

with open(f'outputs/data/validation_summary{suffix}.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSaved summary to outputs/data/validation_summary{suffix}.json')

with open(f'outputs/data/failing_trajectories{suffix}.json', 'w') as f:
    json.dump(failing_trajectories, f)
print(f'Saved {len(failing_trajectories)} failing trajectories to outputs/data/failing_trajectories{suffix}.json')

# plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# plot 1: V0 distribution colored by outcome
ax = axes[0]
inside_cap_V = [outcomes[i]['V0'] for i in inside_captured]
inside_viol_V = [outcomes[i]['V0'] for i in inside_not_captured]
outside_cap_V = [outcomes[i]['V0'] for i in outside_captured]
outside_ok_V = [outcomes[i]['V0'] for i in outside_not_cap]

ax.hist(inside_cap_V, bins=30, alpha=0.6, color='green', label=f'Inside$\\to$captured ({len(inside_cap_V)})')
ax.hist(inside_viol_V, bins=30, alpha=0.6, color='red', label=f'Inside$\\to$NOT captured ({len(inside_viol_V)}) VIOLATIONS')
ax.hist(outside_cap_V, bins=30, alpha=0.6, color='orange', label=f'Outside$\\to$captured ({len(outside_cap_V)})')
ax.hist(outside_ok_V, bins=30, alpha=0.6, color='blue', label=f'Outside$\\to$escaped ({len(outside_ok_V)})')
ax.axvline(0, color='k', linestyle='--', linewidth=2, label='BRT boundary ($V=0$)')
ax.set_xlabel('$V(z_0)$', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Outcome distribution by initial $V$', fontsize=12)
ax.legend(fontsize=7)
ax.grid(True)

# plot 2: capture time histogram for inside BRT trajectories
ax = axes[1]
if capture_times_inside:
    ax.hist(capture_times_inside, bins=30, color='green', alpha=0.8)
    ax.axvline(np.mean(capture_times_inside), color='r', linestyle='--', label=f'Mean: {np.mean(capture_times_inside):.2f}s')
    ax.set_xlabel('Capture time (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Capture time distribution\n(inside BRT trajectories)', fontsize=12)
    ax.legend()
    ax.grid(True)
else:
    ax.text(0.5, 0.5, 'No inside-BRT captures', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Capture time distribution', fontsize=12)

# plot 3: failing trajectories in pz-vz phase plane
ax = axes[2]
dpz_plot = np.linspace(-8, 8, 81)
dvz_plot = np.linspace(-8, 8, 81)
DPZ_p, DVZ_p = np.meshgrid(dpz_plot, dvz_plot)
slice_pts = np.stack([
    np.zeros_like(DPZ_p.ravel()), np.zeros_like(DPZ_p.ravel()), DPZ_p.ravel(),
    np.zeros_like(DPZ_p.ravel()), np.zeros_like(DPZ_p.ravel()), DVZ_p.ravel(),
], axis=1)
V_bg = values_converged_interpolator(slice_pts).reshape(DPZ_p.shape)
ax.pcolormesh(dpz_plot, dvz_plot, V_bg, cmap='RdBu', alpha=0.4, vmin=-3, vmax=3)
ax.contour(dpz_plot, dvz_plot, V_bg, levels=[0], colors='k', linewidths=2)

for ft in failing_trajectories[:50]:
    traj = np.array(ft['trajectory'])
    ax.plot(traj[:, 2], traj[:, 5], 'r-', alpha=0.5, linewidth=0.8)
    ax.plot(traj[0, 2], traj[0, 5], 'r.', markersize=5)

for i in inside_captured[:20]:
    z0 = np.array(outcomes[i]['z0'])
    ax.plot(z0[2], z0[5], 'g.', markersize=4, alpha=0.5)

ax.axvline(-1.0, color='g', linestyle='--', linewidth=1.5)
ax.axvline( 1.0, color='g', linestyle='--', linewidth=1.5)
ax.set_xlabel(r'$\Delta p_z$ (m)', fontsize=12)
ax.set_ylabel(r'$\Delta v_z$ (m/s)', fontsize=12)
ax.set_title(f'Failing trajectories (red) in phase plane\n'
             f'({len(failing_trajectories)} violations, {len(inside_captured)} captures)',
             fontsize=11)
ax.grid(True)

mode_label = 'targeted' if args.sample_inside_only else 'uniform'
fig.suptitle(f'BRT Validation ({mode_label} sampling, n={args.n_samples})', fontsize=13)
fig.tight_layout()
fig.savefig(f'outputs/plots/brt_validation{suffix}.png', bbox_inches='tight')
print(f'Saved outputs/plots/brt_validation{suffix}.png')