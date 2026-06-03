import numpy as np
import jax
import jax.numpy as jnp
import hj_reachability as hj
from hj_reachability import dynamics, sets
from scipy.interpolate import RegularGridInterpolator
import os
import sys
import gc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/data/sweep', exist_ok=True)

GRID_RESOLUTION = (15, 15, 15, 15, 15, 15)
# GRID_RESOLUTION = (21, 21, 21, 21, 21, 21)
# GRID_RESOLUTION = (16, 16, 16, 16, 16, 16)

R_CAPTURE = 1.0
F_E_FIXED  = 2.4
F_P_ALL    = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 5.0]

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
# sample_min = np.array([-10., -10., -10., -10., -10., -10.])
# sample_max = np.array([ 10.,  10.,  10.,  10.,  10.,  10.])

domain_volume = np.prod(sample_max - sample_min)
num_samples = int(1e6)
batch_size  = int(1e4)
num_batches = int(num_samples / batch_size)


class SweepDynamics(dynamics.ControlAndDisturbanceAffineDynamics):
    def __init__(self, f_p, f_e):
        control_space     = sets.Box(jnp.array([-f_p]), jnp.array([f_p]))
        disturbance_space = sets.Box(jnp.array([-f_e]), jnp.array([f_e]))
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


def solve_one(F_P):
    """Solve BRT and compute volume for a single F_P value."""
    ratio = F_P / F_E_FIXED
    print(f'\n=== F_P={F_P:.1f}, F_E={F_E_FIXED:.1f}, ratio={ratio:.3f} ===')

    dyn = SweepDynamics(F_P, F_E_FIXED)
    values = hj.solve(solver_settings, dyn, grid, times, failure_values)
    values_converged = values[-1]

    np.save(f'outputs/data/sweep/values_FP{F_P:.1f}.npy', np.array(values_converged))

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
    print(f'BRT volume: {brt_volume:.2f} m^6  ({100*num_inside/num_samples:.2f}% of domain)')

    # save this result immediately in case of crash
    result = np.array([[ratio, F_P, F_E_FIXED, brt_volume]])
    result_path = f'outputs/data/sweep/result_FP{F_P:.1f}.npy'
    np.save(result_path, result)

    # free GPU memory
    del values
    del values_converged
    jax.clear_caches()
    gc.collect()

    return ratio, F_P, F_E_FIXED, brt_volume


def plot_results():
    """Load all saved per-F_P results and plot."""
    rows = []
    for F_P in F_P_ALL:
        path = f'outputs/data/sweep/result_FP{F_P:.1f}.npy'
        if os.path.exists(path):
            rows.append(np.load(path)[0])
        else:
            print(f'Missing result for F_P={F_P:.1f}, skipping')

    if not rows:
        print('No results found, nothing to plot.')
        return

    results = np.array(rows)
    results = results[results[:, 0].argsort()]  # sort by ratio

    np.save('outputs/data/sweep/sweep_results.npy', results)

    print('\n=== Sweep Results ===')
    print(f'{"Ratio":>8}  {"F_P":>6}  {"F_E":>6}  {"BRT Vol":>12}')
    print('-' * 42)
    for row in results:
        print(f'{row[0]:>8.3f}  {row[1]:>6.1f}  {row[2]:>6.1f}  {row[3]:>12.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    print("DEBUG x values:", results[:, 0].tolist())
    print("DEBUG y values:", results[:, 3].tolist())

    ax = axes[0]
    ax.plot(results[:, 0], results[:, 3], 'bo-', linewidth=2, markersize=8)
    # ax.axvline(1.0, color='r', linestyle='--', linewidth=1.5, label='Equal thrust')
    ax.set_xlabel('Thrust ratio $F_P / F_E$', fontsize=13)
    ax.set_ylabel('BRT Volume (m$^6$)', fontsize=13)
    ax.set_title('BRT Volume vs Thrust Ratio', fontsize=13)
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(results[:, 1], results[:, 3], 'go-', linewidth=2, markersize=8)
    # ax.axvline(F_E_FIXED, color='r', linestyle='--', linewidth=1.5,
    #            label=f'$F_E$ = {F_E_FIXED} N')
    ax.set_xlabel('Pursuer thrust $F_P$ (N)', fontsize=13)
    ax.set_ylabel('BRT Volume (m$^6$)', fontsize=13)
    ax.set_title('BRT Volume vs Pursuer Thrust', fontsize=13)
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig('outputs/plots/thrust_sweep2.png')
    print('Saved outputs/plots/thrust_sweep2.png')


# ── entry point ──
# if called with a float arg, solve that one F_P value
# if called with 'plot', just plot existing results
# if called with no args, run all sequentially (may OOM)
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == 'plot':
        plot_results()
    else:
        F_P = float(arg)
        solve_one(F_P)
else:
    # run all sequentially
    for F_P in F_P_ALL:
        solve_one(F_P)
    plot_results()