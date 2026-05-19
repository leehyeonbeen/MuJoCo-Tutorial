"""Simple locomotion controller — hopper with hand-tuned finite state machine."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco model="hopper">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1"/>
    <geom type="plane" size="10 2 0.1" rgba=".8 .9 .8 1" friction="1.0 0.005 0.0001"/>

    <body name="torso" pos="0 0 1.25">
      <joint name="rootx" type="slide" axis="1 0 0"/>
      <joint name="rootz" type="slide" axis="0 0 1"/>
      <joint name="rooty" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 -0.1 0 0 0.1" size="0.05" mass="3"/>

      <body name="thigh" pos="0 0 -0.1">
        <joint name="thigh_joint" type="hinge" axis="0 1 0"
               range="-150 0" damping="2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="2"/>

        <body name="leg" pos="0 0 -0.4">
          <joint name="leg_joint" type="hinge" axis="0 1 0"
                 range="-150 0" damping="2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" mass="1"/>

          <body name="foot" pos="0 0 -0.4">
            <joint name="foot_joint" type="hinge" axis="0 1 0"
                   range="-45 45" damping="1"/>
            <geom type="capsule" fromto="-0.1 0 0 0.15 0 0" size="0.04" mass="1"
                  friction="2.0 0.005 0.0001"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="thigh_joint" ctrlrange="-50 50" gear="50"/>
    <motor joint="leg_joint" ctrlrange="-50 50" gear="50"/>
    <motor joint="foot_joint" ctrlrange="-20 20" gear="20"/>
  </actuator>
</mujoco>
"""


class HopperFSM:
    """Finite state machine for hopper locomotion.

    States:
      LOADING  — compress legs to store energy
      LAUNCH   — extend legs to push off
      FLIGHT   — retract legs for next landing
      LANDING  — prepare for ground contact
    """

    LOADING, LAUNCH, FLIGHT, LANDING = range(4)

    def __init__(self):
        self.state = self.LOADING
        self.timer = 0.0

    def get_action(self, data, dt):
        height = data.qpos[1]  # rootz
        vel_z = data.qvel[1]   # vertical velocity

        # Detect ground contact (foot geom)
        on_ground = height < 1.0 and vel_z < 0.1

        self.timer += dt
        ctrl = np.zeros(3)

        if self.state == self.LOADING:
            # Crouch: bend thigh and leg
            ctrl[0] = -0.8  # thigh back
            ctrl[1] = -0.6  # leg compress
            ctrl[2] = 0.2   # foot forward lean
            if self.timer > 0.15:
                self.state = self.LAUNCH
                self.timer = 0.0

        elif self.state == self.LAUNCH:
            # Extend: push off ground
            ctrl[0] = 0.9   # thigh forward
            ctrl[1] = 0.9   # leg extend
            ctrl[2] = -0.5  # foot push
            if self.timer > 0.1 or not on_ground:
                self.state = self.FLIGHT
                self.timer = 0.0

        elif self.state == self.FLIGHT:
            # Retract legs, lean forward
            ctrl[0] = -0.3
            ctrl[1] = -0.5
            ctrl[2] = 0.0
            if on_ground and self.timer > 0.1:
                self.state = self.LANDING
                self.timer = 0.0

        elif self.state == self.LANDING:
            # Absorb impact
            ctrl[0] = -0.4
            ctrl[1] = -0.3
            ctrl[2] = 0.1
            if self.timer > 0.05:
                self.state = self.LOADING
                self.timer = 0.0

        return ctrl


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    fsm = HopperFSM()
    dt = model.opt.timestep

    print("Hopper FSM locomotion controller")
    print("Watch the hopper attempt forward hops!")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 20.0:
            ctrl = fsm.get_action(data, dt)
            data.ctrl[:] = ctrl

            mujoco.mj_step(model, data)
            viewer.sync()

        print(f"Final x-position: {data.qpos[0]:.2f} m")


if __name__ == "__main__":
    main()
