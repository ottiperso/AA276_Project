import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os

from dynamics import PursuitEvasion

os.makedirs('outputs', exist_ok=True)

# Grid resolution, can scale up
# coarse:    (11, 11, 11, 11, 11, 11)
# medium: (21, 21, 21, 21, 21, 21)
# fine: (31, 31, 31, 31, 31, 31)
GRID_RESOLUTION = (11, 11, 11, 11, 11, 11)

R_CAPTURE = 1.0

# 6D grid instead of 2D
# state = [delta_px, delta_py, delta_pz, delta_vx, delta_vy, delta_vz]
# same struct as HW2: Box lo/hi, grid resolution
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-5., -5., -5., -5., -5., -5.]),
        np.array([ 5.,  5.,  5.,  5.,  5.,  5.])
    ),
    GRID_RESOLUTION
)

# capture set: ||delta_p|| <= r_capture
# signed distance, negative inside failure (capture) set
# l(z) = ||delta_p|| - r_capture
# negative inside capture set (pursuer wins), positive outside (evader safe)
failure_values = (
    jnp.sqrt(
        grid.states[..., 0]**2 +  # delta_px
        grid.states[..., 1]**2 +  # delta_py
        grid.states[..., 2]**2    # delta_pz
    ) - R_CAPTURE
)

times = np.linspace(0, -5, 101, endpoint=True)
solver_settings = hj.SolverSettings.with_accuracy(
    'very_high',
    hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
)

dynamics_obj = PursuitEvasion()
print(f'Solving BRT with grid resolution {GRID_RESOLUTION}...')
values = hj.solve(solver_settings, dynamics_obj, grid, times, failure_values)

# save outputs (for simulate.py and plot.py to use)
np.save('outputs/values.npy', np.array(values))
np.save('outputs/times.npy', times)
print('Saved BRT to outputs/')

# safe set volume using random sampling
values_converged = values[-1]
values_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(values_converged),
    bounds_error=False,
    fill_value=None
)
num_samples = int(1e5)  # smaller for speed, increase on compute engine
batch_size  = int(1e3)
num_batches = int(num_samples / batch_size)
sample_min = np.array([-5., -5., -5., -5., -5., -5.])
sample_max = np.array([ 5.,  5.,  5.,  5.,  5.,  5.])
num_safe = 0
for _ in tqdm(range(num_batches)):
    samples = np.random.uniform(
        low=sample_min, high=sample_max,
        size=(batch_size, len(sample_min))
    )
    num_safe += np.sum(values_converged_interpolator(samples) > 0)
print(f'BRT Volume (pursuer wins): '
      f'{(num_samples-num_safe)*np.prod(sample_max-sample_min)/num_samples:.3f}')
print(f'Safe Volume (evader wins): '
      f'{num_safe*np.prod(sample_max-sample_min)/num_samples:.3f}')