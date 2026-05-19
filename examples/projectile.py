"""Projectile with drag — modern MuJoCo example."""

import numpy as np
import mujoco
import mujoco.viewer

# Inline MJCF model (no separate XML needed for simple examples)
XML = """
<mujoco>
  <option gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="50 50 0.1" rgba=".9 .9 .9 1"/>
    <body name="ball" pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Launch with initial velocity
    data.qvel[0] = 5.0   # vx
    data.qvel[2] = 10.0  # vz

    # Optional: add drag force in control callback
    drag_coeff = 0.5

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 5.0:
            # Simple drag: F = -c * v
            vel = data.qvel[:3]
            data.qfrc_applied[:3] = -drag_coeff * vel

            mujoco.mj_step(model, data)
            viewer.sync()

    print(f"Final position: {data.qpos[:3]}")


if __name__ == "__main__":
    main()
