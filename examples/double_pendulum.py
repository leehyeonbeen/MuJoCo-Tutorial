"""Double pendulum with energy-shaping control — modern MuJoCo example."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <body name="link1" pos="0 0 2">
      <joint name="joint1" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.04" mass="1" rgba="0.2 0.6 0.8 1"/>
      <body name="link2" pos="0 0 -0.5">
        <joint name="joint2" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" mass="0.5" rgba="0.8 0.3 0.2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="joint1" ctrlrange="-20 20"/>
    <motor joint="joint2" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Target: both joints at 0 (hanging down)
    # Try to swing up: target is pi (upright)
    target = np.array([np.pi, 0.0])
    kp = np.array([100.0, 50.0])
    kd = np.array([20.0, 10.0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 15.0:
            # PD control to target
            error = target - data.qpos[:2]
            data.ctrl[:] = kp * error - kd * data.qvel[:2]
            # Clamp to actuator limits
            data.ctrl[:] = np.clip(data.ctrl, [-20, -10], [20, 10])

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
