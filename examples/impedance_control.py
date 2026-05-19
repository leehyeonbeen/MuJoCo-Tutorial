"""Impedance control — compliant end-effector that yields to external forces."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>

    <!-- Obstacle the arm will push against -->
    <body name="obstacle" pos="0.5 0.0 0.55">
      <geom type="sphere" size="0.05" rgba="1 0.2 0.2 0.8" mass="0.5"
            contype="1" conaffinity="1"/>
      <joint type="slide" axis="1 0 0" damping="5" stiffness="100"/>
    </body>

    <!-- 2-link arm -->
    <body name="link1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" mass="2"
            contype="0" conaffinity="0" rgba="0.3 0.5 0.8 1"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1.5"
              contype="1" conaffinity="1" rgba="0.3 0.8 0.4 1"/>
        <site name="ee" pos="0.3 0 0" size="0.025" rgba="0 0 1 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-100 100"/>
    <motor joint="j2" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    nv = model.nv

    # Desired equilibrium position (into the obstacle)
    x_des = np.array([0.55, 0.0, 0.5])

    # Impedance parameters (spring-damper behavior)
    # Low stiffness = compliant; high stiffness = stiff
    K_imp = np.diag([100.0, 100.0, 100.0])  # stiffness (N/m)
    D_imp = np.diag([20.0, 20.0, 20.0])     # damping (Ns/m)

    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    print("Impedance control demo:")
    print(f"  Stiffness: {np.diag(K_imp)} N/m")
    print(f"  Damping:   {np.diag(D_imp)} Ns/m")
    print("  The arm pushes toward the obstacle but yields on contact.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 20.0:
            mujoco.mj_forward(model, data)

            # Current EE state
            x = data.site_xpos[ee_id].copy()
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
            dx = jacp @ data.qvel

            # Impedance law: F = K*(x_des - x) - D*dx
            # This makes the EE behave like a mass-spring-damper
            F = K_imp @ (x_des - x) - D_imp @ dx

            # Gravity compensation + impedance torque
            tau = jacp.T @ F + data.qfrc_bias
            data.ctrl[:] = np.clip(tau, -100, 100)

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
