import numpy as np
import jax.numpy as jnp
import hj_reachability as hj
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os

from dynamics import PursuitEvasion, F_P_MAX, F_E_MAX

# BRT outputs from solve_brt.py
values = np.load('outputs/data/values.npy')
times  = np.load('outputs/data/times.npy')
values_converged = values[-1]

# same grid as solve_brt.py
# GRID_RESOLUTION = (11, 11, 11, 11, 11, 11)
# GRID_RESOLUTION = (21, 21, 21, 21, 21, 21)
GRID_RESOLUTION = (15, 15, 15, 15, 15, 15) # medium try 2

grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
    # hj.sets.Box(
    #     np.array([-5., -5., -5., -5., -5., -5.]),
    #     np.array([ 5.,  5.,  5.,  5.,  5.,  5.])
    # ),
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

values_converged_interpolator = RegularGridInterpolator(
    ([np.array(v) for v in grid.coordinate_vectors]),
    np.array(values_converged),
    bounds_error=False,
    fill_value=None
)

# gradient of converged value function
grads_converged = grid.grad_values(
    jnp.array(values_converged),
    solver_settings.upwind_scheme
)

# extract deriv wrt relative z velocity, controls only affect rel z vel
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

# pursuer bang-bang: minimize V (opposite sign of control gradient)
# def pursuer_optimal(z):
#     return -np.sign(beta5s_converged_interpolator(z.reshape(1, -1))) * F_P_MAX

def pursuer_optimal(z):
    # z_clipped = np.clip(z, -4.9, 4.9) # to avoid gradient = 0 when outside grid (infinity)
    z_clipped = np.clip(z, -7.9, 7.9)
    grad = beta5s_converged_interpolator(z_clipped.reshape(1, -1)).item()
    if abs(grad) < 1e-6:
        # fallback: thrust toward evader based on z rel posn
        return -np.sign(z[2]) * F_P_MAX if abs(z[2]) > 0.01 else F_P_MAX
    return -np.sign(grad) * F_P_MAX

# evader bang-bang: maximize V (opposite sign of disturbance gradient)
def evader_optimal(z):
    z_clipped = np.clip(z, -7.9, 7.9)
    return -np.sign(beta5s_converged_interpolator(z_clipped.reshape(1, -1))) * F_E_MAX

# CONTROLLERS
def pursuer_control(z):
    return pursuer_optimal(z)       # optimal pursuer
    # return F_P_MAX                # nominal: full thrust up

def evader_control(z):
    return evader_optimal(z)        # optimal evader
    # return 0.0                    # nominal: no thrust

# Euler step for 6D relative dynamics
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

def simulate(z0, nt, dt=0.01):
    zs   = np.full((nt+1, 6), fill_value=np.nan)
    # track individual drone posns separately
    # integrate forward
    p_P  = np.full((nt+1, 3), fill_value=np.nan)
    p_E  = np.full((nt+1, 3), fill_value=np.nan)
    F_Ps = np.full(nt, fill_value=np.nan)
    F_Es = np.full(nt, fill_value=np.nan)

    zs[0] = z0
    # evader at origin, pursuer at z0[:3] (starts at rel posn)
    # evader moves based on F_E
    p_E[0] = np.array([0., 0., 0.])
    p_P[0] = z0[:3]

    # evader velocity
    # evader vel = 0 , pursuer starts at rel velocity
    v_E = np.array([0., 0., 0.])
    # v_P = z0[3:]
    v_P = np.array(z0[3:], dtype=float)

    captured = False
    for i in tqdm(range(nt)):
        z   = zs[i]
        F_P = pursuer_control(z)
        F_E = evader_control(z)

        # debug gradients
        if i % 50 == 0:
            grad_val = beta5s_converged_interpolator(z.reshape(1,-1)).item()
            print(f't={i*dt:.2f} grad={grad_val:.6f} F_P={float(F_P):.2f} dist={np.sqrt(z[0]**2+z[1]**2+z[2]**2):.3f} z={z}')

        # update rel state
        zs[i+1] = euler_step(z, F_P, F_E, dt)
        F_Ps[i] = float(F_P)
        F_Es[i] = float(F_E)

        # update individual drone states
        # evader: only F_E affects vz
        # v_E[2] += float(F_E) * dt
        # p_E[i+1] = p_E[i] + v_E * dt

        # pursuer: only F_P affects vz
        # v_P[2] += float(F_P) * dt
        # p_P[i+1] = p_P[i] + v_P * dt

        p_E[i+1] = p_E[i] + v_E * dt  # position first
        p_P[i+1] = p_P[i] + v_P * dt
        v_E[2] += float(F_E) * dt     # then velocity
        v_P[2] += float(F_P) * dt

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
        print('No capture within time horizon.')
    return zs, p_P, p_E, F_Ps, F_Es

# two initial conditions to test (6D initial states instead of 2D)

# inside BRT: pursuer should capture
z0_inside  = np.array([0., 0., -1.5, 0., 0.,  1.5])  # approaching in z

# outside BRT: evader should escape
z0_outside = np.array([0.,0.,5.,0.,0.,4.])
# z0_outside = np.array([0., 0., 4., 0., 0., 0.])

dt = 0.01
nt = int(8.0 / dt)

zs_in,  p_P_in,  p_E_in,  FPs_in,  FEs_in  = simulate(z0_inside,  nt, dt)
zs_out, p_P_out, p_E_out, FPs_out, FEs_out = simulate(z0_outside, nt, dt)

np.save('outputs/data/zs_inside.npy',   zs_in)
np.save('outputs/data/zs_outside.npy',  zs_out)
np.save('outputs/data/p_P_inside.npy',  p_P_in)
np.save('outputs/data/p_P_outside.npy', p_P_out)
np.save('outputs/data/p_E_inside.npy',  p_E_in)
np.save('outputs/data/p_E_outside.npy', p_E_out)
np.save('outputs/data/FPs_inside.npy',  FPs_in)
np.save('outputs/data/FPs_outside.npy', FPs_out)
np.save('outputs/data/FEs_inside.npy',  FEs_in)
np.save('outputs/data/FEs_outside.npy', FEs_out)
print('Saved trajectories to outputs/data/')

# V at initial conditions
V_inside = values_converged_interpolator(z0_inside.reshape(1,-1))
V_outside = values_converged_interpolator(z0_outside.reshape(1,-1))
print(f'V at z0_inside: {V_inside}')   # should be < 0 if inside BRT
print(f'V at z0_outside: {V_outside}') # should be > 0 if outside BRT

# print(f'min V: {np.array(values_converged).min():.3f}')
# print(f'max V: {np.array(values_converged).max():.3f}')

# find grid points where V < 0 (inside BRT)
coords = np.array(np.meshgrid(*[np.array(v) for v in grid.coordinate_vectors], indexing='ij')).reshape(6, -1).T
V_flat = np.array(values_converged).ravel()
inside_brt = coords[V_flat < 0]
print(f'Number of BRT points: {len(inside_brt)}')
print(f'Sample BRT points:\n{inside_brt[:5]}')

dist_min = np.sqrt(zs_in[:,0]**2 + zs_in[:,1]**2 + zs_in[:,2]**2).min()
print(f'Minimum distance reached: {dist_min:.3f} m (capture radius: 1.0 m)')