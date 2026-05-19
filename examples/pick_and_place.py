"""Pick-and-place with a simple gripper — multi-body coordination example."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <default>
    <joint damping="5"/>
    <geom contype="1" conaffinity="1" friction="1 0.005 0.001"/>
  </default>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".85 .85 .85 1"/>

    <!-- Object to pick -->
    <body name="box" pos="0.3 0.0 0.025">
      <freejoint name="box_joint"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.1" rgba="1 0.3 0 1"/>
    </body>

    <!-- Place target (visual only) -->
    <site name="place_target" pos="-0.3 0.0 0.025" size="0.03" rgba="0 1 0 0.3"/>

    <!-- Simple planar arm with gripper -->
    <body name="base" pos="0 0 0.3">
      <joint name="j_shoulder" type="hinge" axis="0 1 0" range="-180 180"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="1" rgba="0.4 0.4 0.7 1"/>

      <body name="forearm" pos="0.25 0 0">
        <joint name="j_elbow" type="hinge" axis="0 1 0" range="-150 150"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.015" mass="0.7" rgba="0.4 0.7 0.4 1"/>

        <!-- Wrist -->
        <body name="wrist" pos="0.2 0 0">
          <joint name="j_wrist" type="hinge" axis="0 1 0" range="-90 90"/>
          <geom type="cylinder" size="0.02 0.01" rgba="0.5 0.5 0.5 1" mass="0.2"/>

          <!-- Gripper fingers -->
          <body name="finger_l" pos="0.0 0.015 0">
            <joint name="j_finger_l" type="slide" axis="0 1 0" range="0 0.03" damping="20"/>
            <geom type="box" size="0.01 0.005 0.03" pos="0 0 -0.03" rgba="0.7 0.3 0.3 1" mass="0.05"/>
          </body>
          <body name="finger_r" pos="0.0 -0.015 0">
            <joint name="j_finger_r" type="slide" axis="0 -1 0" range="0 0.03" damping="20"/>
            <geom type="box" size="0.01 0.005 0.03" pos="0 0 -0.03" rgba="0.7 0.3 0.3 1" mass="0.05"/>
          </body>

          <site name="grip_site" pos="0 0 -0.035" size="0.005" rgba="0 0 1 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position joint="j_shoulder" kp="100" ctrlrange="-3.14 3.14"/>
    <position joint="j_elbow" kp="80" ctrlrange="-2.6 2.6"/>
    <position joint="j_wrist" kp="40" ctrlrange="-1.57 1.57"/>
    <position joint="j_finger_l" kp="500" ctrlrange="0 0.03"/>
    <position joint="j_finger_r" kp="500" ctrlrange="0 0.03"/>
  </actuator>
</mujoco>
"""


class PickAndPlaceController:
    """Waypoint-based pick-and-place state machine."""

    def __init__(self):
        # Joint targets for key poses (shoulder, elbow, wrist, finger_l, finger_r)
        self.phases = [
            # Phase 0: Open gripper, move above object
            {"joints": [0.6, -1.2, 0.6], "gripper": 0.03, "duration": 1.5},
            # Phase 1: Lower to grasp
            {"joints": [0.8, -1.4, 0.6], "gripper": 0.03, "duration": 1.0},
            # Phase 2: Close gripper
            {"joints": [0.8, -1.4, 0.6], "gripper": 0.0, "duration": 0.8},
            # Phase 3: Lift
            {"joints": [0.4, -0.8, 0.4], "gripper": 0.0, "duration": 1.2},
            # Phase 4: Move to place location
            {"joints": [-0.6, -1.0, 0.6], "gripper": 0.0, "duration": 1.5},
            # Phase 5: Lower to place
            {"joints": [-0.8, -1.4, 0.6], "gripper": 0.0, "duration": 1.0},
            # Phase 6: Release
            {"joints": [-0.8, -1.4, 0.6], "gripper": 0.03, "duration": 0.8},
            # Phase 7: Retract
            {"joints": [0.0, -0.5, 0.0], "gripper": 0.03, "duration": 1.5},
        ]
        self.phase = 0
        self.timer = 0.0

    def get_ctrl(self, dt):
        if self.phase >= len(self.phases):
            self.phase = len(self.phases) - 1

        p = self.phases[self.phase]
        self.timer += dt

        ctrl = np.zeros(5)
        ctrl[0:3] = p["joints"]
        ctrl[3] = p["gripper"]
        ctrl[4] = p["gripper"]

        if self.timer >= p["duration"]:
            self.timer = 0.0
            self.phase += 1
            if self.phase < len(self.phases):
                print(f"  Phase {self.phase}: {['above', 'lower', 'grasp', 'lift', 'move', 'place', 'release', 'retract'][self.phase]}")

        return ctrl


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    controller = PickAndPlaceController()
    dt = model.opt.timestep

    print("Pick-and-place demo")
    print("  Phase 0: moving above object")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 12.0:
            data.ctrl[:] = controller.get_ctrl(dt)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
