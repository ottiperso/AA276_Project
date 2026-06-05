# boundary_sim.py
# Finds and simulates trajectories from near boundary ICs

import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os

from dynamics import PursuitEvasion, F_P_MAX, F_E_MAX

os.makedirs('outputs/data', exist_ok=True)

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
solver_settings = hj.SolverSettings.with_accuracy('very_high',
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

def euler_step(z, F_P, F_E, dt=0.01):
    dzdt = np.array([z[3], z[4], z[5], 0., 0., float(F_P) - float(F_E)])
    return z + dt * dzdt

def simulate(z0, nt, dt=0.01):
    zs  = np.full((nt+1, 6), fill_value=np.nan)
    p_P = np.full((nt+1, 3), fill_value=np.nan)
    p_E = np.full((nt+1, 3), fill_value=np.nan)
    F_Ps = np.full(nt, fill_value=np.nan)
    F_Es = np.full(nt, fill_value=np.nan)

    zs[0]  = z0
    p_E[0] = np.array([0., 0., 0.])
    p_P[0] = z0[:3]
    v_E    = np.array([0., 0., 0.])
    v_P    = np.array(z0[3:], dtype=float)

    captured = False
    for i in tqdm(range(nt)):
        z   = zs[i]
        F_P = pursuer_control(z)
        F_E = evader_control(z)

        zs[i+1] = euler_step(z, F_P, F_E, dt)
        F_Ps[i] = float(F_P)
        F_Es[i] = float(F_E)

        p_E[i+1] = p_E[i] + v_E * dt
        p_P[i+1] = p_P[i] + v_P * dt
        v_E[2]  += float(F_E) * dt
        v_P[2]  += float(F_P) * dt

        dist = np.sqrt(z[0]**2 + z[1]**2 + z[2]**2)
        if dist <= 1.0:
            print(f'Capture at t={i*dt:.2f}s!')
            captured = True
            zs   = zs[:i+1]
            p_P  = p_P[:i+1]
            p_E  = p_E[:i+1]
            F_Ps = F_Ps[:i]
            F_Es = F_Es[:i]
            break

    if not captured:
        print('No capture.')
    return zs, p_P, p_E, F_Ps, F_Es

# find boundary or near boundary pts 
# scan pz axis at dvz=0
# print('\nScanning pz axis at dvz=0 for boundary...')
for dpz in np.linspace(0.5, 5.0, 50):
    z_test = np.array([0., 0., dpz, 0., 0., 0.])
    V = values_converged_interpolator(z_test.reshape(1,-1)).item()
    if abs(V) < 0.15:
        print(f'Near boundary: dpz={dpz:.3f}  V={V:.4f}')

# scan dvz axis at dpz=0
# print('\nScanning dvz axis at dpz=0 for boundary...')
for dvz in np.linspace(-5.0, 5.0, 100):
    z_test = np.array([0., 0., 0., 0., 0., dvz])
    V = values_converged_interpolator(z_test.reshape(1,-1)).item()
    if abs(V) < 0.15:
        print(f'Near boundary: dvz={dvz:.3f}  V={V:.4f}')

# after scanning, chosen:
boundary_ics = {
    'boundary_pz_pos':  np.array([0., 0.,  2.0, 0., 0.,  0.0]),  # pz>0, dvz=0
    'boundary_pz_neg':  np.array([0., 0., -2.0, 0., 0.,  0.0]),  # pz<0, dvz=0
    'boundary_pz_other':   np.array([0., 0.,  3.056, 0., 0.,  0.000]),  # V=+0.0025
    'boundary_dvz_other':  np.array([0., 0.,  2.000, 0., 0.,  0.455]),  # V=+0.0084
    'boundary_diag': np.array([0., 0.,  1.545, 0., 0.,  0.773]),  # V=+0.0048
}

dt = 0.01
nt = int(8.0 / dt)

print('\nV values at boundary ICs:')
for name, z0 in boundary_ics.items():
    V0 = values_converged_interpolator(z0.reshape(1,-1)).item()
    print(f'{name:25s}: V={V0:+.4f}  z0={z0}')

print('\nSimulating boundary ICs...')
for name, z0 in boundary_ics.items():
    print(f'\n{name}')
    zs, p_P, p_E, FPs, FEs = simulate(z0, nt, dt)
    np.save(f'outputs/data/zs_{name}.npy',   zs)
    np.save(f'outputs/data/p_P_{name}.npy',  p_P)
    np.save(f'outputs/data/p_E_{name}.npy',  p_E)
    np.save(f'outputs/data/FPs_{name}.npy',  FPs)
    np.save(f'outputs/data/FEs_{name}.npy',  FEs)
    V0 = values_converged_interpolator(z0.reshape(1,-1)).item()
    print(f'V(z0) = {V0:.4f}')

# print('\nDone! Run animations with:')
# for name in boundary_ics:
#     print(f'python animate_brt.py --ic {name} --skip 3 --fps 30')