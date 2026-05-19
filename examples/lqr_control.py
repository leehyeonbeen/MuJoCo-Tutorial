"""LQR (Linear Quadratic Regulator) for cart-pole balancing."""

import numpy as np
import scipy.linalg
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>

    <!-- Cart on a rail -->
    <body name="cart" pos="0 0 0.1">
      <joint name="cart_slide" type="slide" axis="1 0 0" damping="0.1"/>
      <geom type="box" size="0.1 0.1 0.05" mass="5" rgba="0.2 0.2 0.8 1"/>

      <!-- Pole -->
      <body name="pole" pos="0 0 0.05">
        <joint name="pole_hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="1" rgba="0.9 0.3 0.1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="cart_slide" ctrlrange="-20 20" gear="1"/>
  </actuator>
</mujoco>
"""


def linearize(model, data, eps=1e-6):
    """Finite-difference linearization around the current state (upright equilibrium)."""
    nv = model.nv
    nu = model.nu
    nx = 2 * nv  # state = [qpos, qvel]

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    # Save nominal state
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    ctrl0 = data.ctrl.copy()

    # Compute nominal next state
    mujoco.mj_step(model, data)
    x_next_nom = np.concatenate([data.qpos.copy(), data.qvel.copy()])

    # Linearize w.r.t. state
    for i in range(nx):
        # Reset
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        data.ctrl[:] = ctrl0

        # Perturb
        if i < nv:
            data.qpos[i] += eps
        else:
            data.qvel[i - nv] += eps

        mujoco.mj_step(model, data)
        x_next_pert = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        A[:, i] = (x_next_pert - x_next_nom) / eps

    # Linearize w.r.t. control
    for i in range(nu):
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        data.ctrl[:] = ctrl0
        data.ctrl[i] += eps

        mujoco.mj_step(model, data)
        x_next_pert = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        B[:, i] = (x_next_pert - x_next_nom) / eps

    # Restore
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.ctrl[:] = ctrl0

    return A, B


def compute_lqr_gain(A, B, Q, R):
    """Solve discrete-time algebraic Riccati equation for LQR gain K."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    nv = model.nv
    nx = 2 * nv

    # Set to upright equilibrium: pole angle = 0 (already default)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    # Linearize around equilibrium
    A, B = linearize(model, data)

    # LQR cost matrices
    Q = np.diag([1.0, 10.0, 0.1, 1.0])  # [cart_pos, pole_angle, cart_vel, pole_vel]
    R = np.array([[0.01]])

    K = compute_lqr_gain(A, B, Q, R)
    print(f"LQR gain K = {K}")

    # Reset with small perturbation
    data.qpos[:] = [0.0, 0.1]  # slight pole tilt
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 15.0:
            # State vector
            x = np.concatenate([data.qpos, data.qvel])

            # LQR control: u = -K * x
            u = -K @ x
            data.ctrl[0] = np.clip(u[0], -20, 20)

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
