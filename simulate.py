import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os

from dynamics import PursuitEvasion, F_P_MAX, F_E_MAX

# BRT outputs from solve_brt.py
values = np.load('outputs/values.npy')
times  = np.load('outputs/times.npy')
values_converged = values[-1]

# same grid as solve_brt.py
GRID_RESOLUTION = (11, 11, 11, 11, 11, 11)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    hj.sets.Box(
        np.array([-5., -5., -5., -5., -5., -5.]),
        np.array([ 5.,  5.,  5.,  5.,  5.,  5.])
    ),
    GRID_RESOLUTION
)
solver_settings = hj.SolverSettings.with_accuracy(
    'very_high',
    hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
)

# compute gradient of converged value function
grads_converged = grid.grad_values(
    jnp.array(values_converged),
    solver_settings.upwind_scheme
)
# 6D gradient, index 5 = d/d(delta_vz)
# hw2: beta2s = grads[:,:,1]
beta5s_converged = grads_converged[..., 5]

# interpolator for gradient
beta5s_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(beta5s_converged),
    bounds_error=False,
    fill_value=None
)

# pursuer bang-bang: minimize V -> sign of dV/d(delta_vz)
def pursuer_optimal(z):
    return -np.sign(beta5s_converged_interpolator(z.reshape(1, -1))) * F_P_MAX

# evader bang-bang: maximize V -> opposite sign
def evader_optimal(z):
    return np.sign(beta5s_converged_interpolator(z.reshape(1, -1))) * F_E_MAX

# CONTROLLER
def pursuer_control(z):
    return pursuer_optimal(z)       # optimal pursuer
    # return F_P_MAX                # nominal: full thrust up

def evader_control(z):
    return evader_optimal(z)        # optimal evader
    # return 0.0                    # nominal: no thrust

# Euler Step for 6D relative dynamics
def euler_step(z, F_P, F_E, dt=0.01):
    dzdt = np.array([
        z[3],           # d(delta_px)/dt = delta_vx
        z[4],           # d(delta_py)/dt = delta_vy
        z[5],           # d(delta_pz)/dt = delta_vz
        0.,             # d(delta_vx)/dt = 0 (near-hover)
        0.,             # d(delta_vy)/dt = 0 (near-hover)
        float(F_P) - float(F_E)  # d(delta_vz)/dt = F_P - F_E
    ])
    return z + dt * dzdt

# hw2: simulate(x0, nt, dt)
def simulate(z0, nt, dt=0.01):
    zs  = np.full((nt+1, 6), fill_value=np.nan)
    F_Ps = np.full(nt, fill_value=np.nan)
    F_Es = np.full(nt, fill_value=np.nan)
    zs[0] = z0
    captured = False
    for i in tqdm(range(nt)):
        z    = zs[i]
        F_P  = pursuer_control(z)
        F_E  = evader_control(z)
        zs[i+1] = euler_step(z, F_P, F_E, dt)
        F_Ps[i] = float(F_P)
        F_Es[i] = float(F_E)
        # check capture condition
        dist = np.sqrt(z[0]**2 + z[1]**2 + z[2]**2)
        if dist <= 1.0:
            print(f'Capture at t={i*dt:.2f}s!')
            captured = True
            zs   = zs[:i+1]
            F_Ps = F_Ps[:i]
            F_Es = F_Es[:i]
            break
    if not captured:
        print('No capture within time horizon.')
    return zs, F_Ps, F_Es

# two initial conditions to test (6D initial states instead of 2D)
# inside BRT: pursuer should capture
z0_inside  = np.array([1.5, 0., 0., 0., 0., 0.])
# outside BRT: evader should escape
z0_outside = np.array([4.0, 0., 0., 0., 0., 4.0])

dt = 0.01
nt = int(5.0 / dt)

print('Simulating from inside BRT (capture expected)...')
zs_in,  FPs_in,  FEs_in  = simulate(z0_inside,  nt, dt)

print('Simulating from outside BRT (escape expected)...')
zs_out, FPs_out, FEs_out = simulate(z0_outside, nt, dt)

np.save('outputs/zs_inside.npy',   zs_in)
np.save('outputs/zs_outside.npy',  zs_out)
np.save('outputs/FPs_inside.npy',  FPs_in)
np.save('outputs/FPs_outside.npy', FPs_out)
np.save('outputs/FEs_inside.npy',  FEs_in)
np.save('outputs/FEs_outside.npy', FEs_out)
print('Saved trajectories to outputs/')