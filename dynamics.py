# dynamics.py
# defines the 6D relative dynamics class for HJ reachability

import jax.numpy as jnp
import hj_reachability as hj
from hj_reachability import dynamics, sets

# Control bounds (pursuer more capable than evader)
F_P_MAX = 2.50  # pursuer thrust bound
F_E_MAX = 2.40  # evader thrust bound

class PursuitEvasion(dynamics.ControlAndDisturbanceAffineDynamics):
    # 6D relative state (instead of 2D pendulum)
    # state z = [delta_px, delta_py, delta_pz, delta_vx, delta_vy, delta_vz]
    # control = pursuer thrust F_P (1D)
    # disturbance = evader thrust F_E (1D, adversarial)

    def __init__(self):
        # control/disturbance bounds: pursuer/evader thrusts
        control_space = sets.Box(
            jnp.array([-F_P_MAX]),
            jnp.array([ F_P_MAX])
        )
        disturbance_space = sets.Box(
            jnp.array([-F_E_MAX]),
            jnp.array([ F_E_MAX])
        )
        super().__init__(
            control_mode='min',      # pursuer minimizes V (wants capture, V<=0)
            disturbance_mode='max',  # evader maximizes V (wants escape, V>0)
            control_space=control_space,
            disturbance_space=disturbance_space
        )

    def open_loop_dynamics(self, state, time):
        # 6D relative dynamics under near-hover assumption
        # delta_p_dot = delta_v
        # delta_v_dot = 0 for x,y (no lateral thrust under near-hover)
        # delta_vz_dot = F_P - F_E (handled by control/disturbance jacobians)
        _, _, _, dvx, dvy, dvz = state
        return jnp.array([dvx, dvy, dvz, 0., 0., 0.])

    def control_jacobian(self, state, time):
        # F_P enters delta_vz (index 5) (1 * F_P * u)
        return jnp.array([[0.], [0.], [0.], [0.], [0.], [1.]])

    def disturbance_jacobian(self, state, time):
        # F_E enters delta_vz (index 5) (1 * F_E * d)
        # evader opposes pursuer in vertical direction
        return jnp.array([[0.], [0.], [0.], [0.], [0.], [-1.]])