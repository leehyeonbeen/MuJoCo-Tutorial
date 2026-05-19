"""2D Hopper with simple jumping controller — modern MuJoCo example."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco model="hopper">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 -2 3" dir="0 0.5 -1"/>
    <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>

    <!-- Torso -->
    <body name="torso" pos="0 0 1.25">
      <joint name="rootx" type="slide" axis="1 0 0"/>
      <joint name="rootz" type="slide" axis="0 0 1"/>
      <joint name="rooty" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 -0.2 0 0 0.2" size="0.05" mass="3" rgba="0.3 0.5 0.7 1"/>

      <!-- Thigh -->
      <body name="thigh" pos="0 0 -0.2">
        <joint name="thigh_joint" type="hinge" axis="0 1 0" range="-150 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.45" size="0.04" mass="2" rgba="0.4 0.6 0.3 1"/>

        <!-- Leg -->
        <body name="leg" pos="0 0 -0.45">
          <joint name="leg_joint" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" mass="1" rgba="0.6 0.3 0.4 1"/>

          <!-- Foot -->
          <body name="foot" pos="0 0 -0.5">
            <joint name="foot_joint" type="hinge" axis="0 1 0" range="-45 45"/>
            <geom type="capsule" fromto="-0.1 0 0 0.15 0 0" size="0.03" mass="0.5" rgba="0.7 0.7 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="thigh_joint" ctrlrange="-100 100" gear="100"/>
    <motor joint="leg_joint" ctrlrange="-100 100" gear="100"/>
    <motor joint="foot_joint" ctrlrange="-100 100" gear="50"/>
  </actuator>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Simple open-loop hopping pattern
    phase = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 10.0:
            phase = data.time * 4.0  # 4 Hz hopping

            # Sinusoidal joint commands for hopping
            data.ctrl[0] = -0.8 * np.sin(phase)  # thigh
            data.ctrl[1] = -0.5 * np.sin(phase + 0.5)  # leg
            data.ctrl[2] = 0.3 * np.cos(phase)  # foot

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
