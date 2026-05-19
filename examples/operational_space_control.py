"""Operational space control — task-space dynamics compensation for a robot arm."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>

    <!-- Target (visual) -->
    <site name="target" pos="0.4 0.0 0.8" size="0.025" rgba="1 0 0 0.5"/>

    <!-- 3-DOF arm with realistic inertia -->
    <body name="link1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" mass="2" rgba="0.3 0.3 0.8 1"/>
      <body name="link2" pos="0 0 0.3">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" mass="1.5" rgba="0.3 0.7 0.3 1"/>
        <body name="link3" pos="0.3 0 0">
          <joint name="j3" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="1" rgba="0.8 0.5 0.2 1"/>
          <site name="end_effector" pos="0.25 0 0" size="0.02" rgba="0 0 1 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-100 100"/>
    <motor joint="j2" ctrlrange="-100 100"/>
    <motor joint="j3" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target")
    nv = model.nv

    # Gains
    kp = np.diag([600.0, 600.0, 600.0])
    kd = np.diag([50.0, 50.0, 50.0])

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 15.0:
            # Forward kinematics
            mujoco.mj_forward(model, data)

            # Positions
            x = data.site_xpos[ee_id].copy()
            x_des = data.site_xpos[target_id].copy()

            # Jacobian
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
            J = jacp  # position Jacobian only

            # End-effector velocity
            dx = J @ data.qvel

            # Joint-space inertia matrix
            M = np.zeros((nv, nv))
            mujoco.mj_fullM(model, M, data.qM)

            # Task-space inertia: Lambda = (J M^-1 J^T)^-1
            M_inv = np.linalg.inv(M)
            Lambda = np.linalg.inv(J @ M_inv @ J.T + 1e-6 * np.eye(3))

            # Coriolis + gravity compensation
            h = data.qfrc_bias.copy()  # C(q,dq)*dq + g(q)

            # Task-space PD force
            F = kp @ (x_des - x) - kd @ dx

            # Operational space control law:
            # tau = J^T * Lambda * F + h
            tau = J.T @ (Lambda @ F) + h
            data.ctrl[:] = np.clip(tau, -100, 100)

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
