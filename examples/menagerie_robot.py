"""Loading a robot from MuJoCo Menagerie and performing basic control."""

import numpy as np
import mujoco
import mujoco.viewer


def main():
    """Load Franka Panda from Menagerie and do joint-space PD control.

    Prerequisites:
        git clone https://github.com/google-deepmind/mujoco_menagerie
    """
    # Try loading from menagerie (adjust path as needed)
    menagerie_path = "mujoco_menagerie/franka_emika_panda/panda.xml"

    try:
        model = mujoco.MjModel.from_xml_path(menagerie_path)
    except Exception:
        print(f"Could not load {menagerie_path}")
        print("Please clone mujoco_menagerie first:")
        print("  git clone https://github.com/google-deepmind/mujoco_menagerie")
        return

    data = mujoco.MjData(model)

    # Get number of actuators
    nu = model.nu
    print(f"Loaded Franka Panda: {model.nq} DOFs, {nu} actuators")

    # Home position
    home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])

    # Target position (reaching to the side)
    target = np.array([0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5])

    # PD gains
    kp = np.ones(nu) * 100.0
    kd = np.ones(nu) * 20.0

    # Initialize at home
    data.qpos[:7] = home

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 5.0:
            # Interpolate target over time
            t = min(data.time / 2.0, 1.0)  # ramp over 2 seconds
            current_target = home + t * (target - home)

            # PD control (first 7 joints are arm, rest may be gripper)
            q = data.qpos[:7]
            qd = data.qvel[:7]
            tau = kp[:7] * (current_target - q) - kd[:7] * qd
            data.ctrl[:7] = tau

            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
