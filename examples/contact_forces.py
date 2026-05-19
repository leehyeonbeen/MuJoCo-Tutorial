"""Contact detection and force sensing — modern MuJoCo example."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba=".8 .8 .8 1"/>

    <!-- Falling box -->
    <body name="box" pos="0 0 1.5">
      <joint type="free"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1" rgba="0.8 0.2 0.2 1"/>
    </body>

    <!-- Table surface -->
    <body name="table" pos="0 0 0.5">
      <geom name="table_top" type="box" size="0.5 0.5 0.02" rgba="0.5 0.4 0.3 1"/>
    </body>
  </worldbody>

  <!-- Force sensor on the box -->
  <sensor>
    <touch name="box_touch" site="box_site"/>
  </sensor>

  <!-- Need a site on the box for the touch sensor -->
  <worldbody>
    <body name="box2" pos="0 0 1.5">
      <joint type="free"/>
      <geom name="box2_geom" type="box" size="0.1 0.1 0.1" mass="1" rgba="0.2 0.2 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Simpler version without sensor complications
SIMPLE_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba=".8 .8 .8 1"/>

    <body name="box" pos="0 0 2">
      <joint type="free"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" mass="1" rgba="0.8 0.2 0.2 1"/>
    </body>

    <body name="table" pos="0 0 0.5">
      <geom name="table_top" type="box" size="0.5 0.5 0.02" rgba="0.5 0.4 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(SIMPLE_XML)
    data = mujoco.MjData(model)

    print("Simulating box dropping onto table...")
    print(f"{'Time':>6s} {'Height':>8s} {'Contacts':>10s} {'Force':>10s}")
    print("-" * 40)

    for step in range(2000):
        mujoco.mj_step(model, data)

        # Check contacts every 100 steps
        if step % 100 == 0:
            # Read contact information
            n_contacts = data.ncon
            total_force = 0.0

            for i in range(n_contacts):
                contact = data.contact[i]
                # Get contact force
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                total_force += np.linalg.norm(force[:3])

            box_height = data.qpos[2]
            print(f"{data.time:6.3f} {box_height:8.4f} {n_contacts:10d} {total_force:10.2f}")

    print("\nDone! The box has settled on the table.")
    print(f"Final box height: {data.qpos[2]:.4f}")


if __name__ == "__main__":
    main()
