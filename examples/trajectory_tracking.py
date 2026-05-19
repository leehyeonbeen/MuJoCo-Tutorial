"""Trajectory tracking — 3-link arm follows a circular path in task space."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>

    <!-- 3-link planar arm -->
    <body name="link1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" mass="1" rgba="0.2 0.6 0.9 1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.8" rgba="0.3 0.8 0.4 1"/>
        <body name="link3" pos="0.3 0 0">
          <joint name="j3" type="hinge" axis="0 1 0" damping="0.3"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.015" mass="0.5" rgba="0.9 0.5 0.1 1"/>
          <site name="end_effector" pos="0.2 0 0" size="0.015" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-50 50"/>
    <motor joint="j2" ctrlrange="-50 50"/>
    <motor joint="j3" ctrlrange="-50 50"/>
  </actuator>
</mujoco>
"""


def circle_trajectory(t, center=(0.4, 0.0, 0.6), radius=0.15, freq=0.3):
    """Generate a circular reference trajectory in the XZ plane."""
    cx, cy, cz = center
    angle = 2 * np.pi * freq * t
    x = cx + radius * np.cos(angle)
    z = cz + radius * np.sin(angle)
    # Velocity (derivative)
    dx = -radius * 2 * np.pi * freq * np.sin(angle)
    dz = radius * 2 * np.pi * freq * np.cos(angle)
    return np.array([x, cy, z]), np.array([dx, 0.0, dz])


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    nv = model.nv

    # Control gains
    kp = 200.0
    kd = 40.0

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 20.0:
            # Desired position and velocity on the circle
            pos_des, vel_des = circle_trajectory(data.time)

            # Current end-effector position
            pos_curr = data.site_xpos[ee_site_id].copy()

            # Compute Jacobian
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

            # Current end-effector velocity: J * qdot
            vel_curr = jacp @ data.qvel

            # Task-space PD control: F = kp*(x_des - x) + kd*(dx_des - dx)
            force_task = kp * (pos_des - pos_curr) + kd * (vel_des - vel_curr)

            # Map to joint torques: tau = J^T * F
            data.ctrl[:] = jacp.T @ force_task

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
