"""PD control of a simple pendulum — modern MuJoCo example."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <body name="pendulum" pos="0 0 2">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.04" mass="1" rgba="0 .7 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" name="torque" gear="1" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # PD gains
    kp, kd = 50.0, 10.0
    target_angle = np.pi / 4  # 45 degrees

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 10.0:
            # PD controller
            angle = data.qpos[0]
            velocity = data.qvel[0]
            data.ctrl[0] = kp * (target_angle - angle) - kd * velocity

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
