import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from dynamics import PursuitEvasion

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/data/sweep', exist_ok=True)

GRID_RESOLUTION = (15, 15, 15, 15, 15, 15)
R_CAPTURE = 1.0

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-8., -8., -8., -8., -8., -8.]),
        np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
    ),
    GRID_RESOLUTION
)

failure_values = (
    jnp.sqrt(
        grid.states[..., 0]**2 +
        grid.states[..., 1]**2 +
        grid.states[..., 2]**2
    ) - R_CAPTURE
)

times = np.linspace(0, -10, 101, endpoint=True)
solver_settings = hj.SolverSettings.with_accuracy(
    'very_high',
    hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
)

sample_min = np.array([-8., -8., -8., -8., -8., -8.])
sample_max = np.array([ 8.,  8.,  8.,  8.,  8.,  8.])
domain_volume = np.prod(sample_max - sample_min)
num_samples = int(1e5)
batch_size  = int(1e3)
num_batches = int(num_samples / batch_size)

# thrust ratios to sweep: fix F_E_MAX=2.4, vary F_P_MAX
F_E_FIXED = 2.4
F_P_VALUES = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 5.0]
# also do equal and pursuer-weaker cases
# F_P < F_E means evader always wins → BRT volume = 0 theoretically

results = []

for F_P in F_P_VALUES:
    print(f'\n=== F_P={F_P:.1f}, F_E={F_E_FIXED:.1f}, ratio={F_P/F_E_FIXED:.3f} ===')

    # patch dynamics with new thrust values
    from hj_reachability import dynamics, sets
    class SweepDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
        def __init__(self):
            control_space = sets.Box(
                jnp.array([-F_P]),
                jnp.array([ F_P])
            )
            disturbance_space = sets.Box(
                jnp.array([-F_E_FIXED]),
                jnp.array([ F_E_FIXED])
            )
            super().__init__(
                control_mode='min',
                disturbance_mode='max',
                control_space=control_space,
                disturbance_space=disturbance_space
            )
        def open_loop_dynamics(self, state, time):
            _, _, _, dvx, dvy, dvz = state
            return jnp.array([dvx, dvy, dvz, 0., 0., 0.])
        def control_jacobian(self, state, time):
            return jnp.array([[0.],[0.],[0.],[0.],[0.],[1.]])
        def disturbance_jacobian(self, state, time):
            return jnp.array([[0.],[0.],[0.],[0.],[0.],[-1.]])

    dyn = SweepDynamics()
    values = hj.solve(solver_settings, dyn, grid, times, failure_values)
    values_converged = values[-1]

    np.save(f'outputs/data/sweep/values_FP{F_P:.1f}.npy', np.array(values_converged))

    # compute BRT volume by sampling
    interp = RegularGridInterpolator(
        ([np.array(v) for v in grid.coordinate_vectors]),
        np.array(values_converged),
        bounds_error=False,
        fill_value=None
    )
    num_inside = 0
    for _ in range(num_batches):
        samples = np.random.uniform(low=sample_min, high=sample_max,
                                    size=(batch_size, 6))
        num_inside += np.sum(interp(samples) < 0)

    brt_volume = num_inside * domain_volume / num_samples
    ratio = F_P / F_E_FIXED
    results.append((ratio, F_P, brt_volume))
    print(f'BRT volume: {brt_volume:.2f} m^6')

# save results table
results = np.array(results)
np.save('outputs/data/sweep/sweep_results.npy', results)
print('\nSweep results (ratio, F_P, BRT_volume):')
print(results)

# plot BRT volume vs thrust ratio
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results[:, 0], results[:, 2], 'bo-', linewidth=2, markersize=8)
ax.axvline(1.0, color='r', linestyle='--', linewidth=1.5, label='Equal thrust (ratio=1)')
ax.set_xlabel('Thrust ratio $F_P / F_E$', fontsize=13)
ax.set_ylabel('BRT Volume (m$^6$)', fontsize=13)
ax.set_title('BRT Volume vs Pursuer/Evader Thrust Ratio', fontsize=13)
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig('outputs/plots/thrust_sweep.png')
print('Saved thrust_sweep.png')