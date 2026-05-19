"""Record and playback trajectories — save/load simulation states."""

import numpy as np
import mujoco
import mujoco.viewer

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>

    <body name="arm" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1" rgba="0.3 0.6 0.9 1"/>
      <body pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.015" mass="0.7" rgba="0.4 0.8 0.3 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-10 10"/>
    <motor joint="j2" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def record_trajectory(model, data, duration=5.0):
    """Record a trajectory with sinusoidal control inputs."""
    dt = model.opt.timestep
    n_steps = int(duration / dt)

    # Storage: time, qpos, qvel, ctrl
    trajectory = {
        "time": np.zeros(n_steps),
        "qpos": np.zeros((n_steps, model.nq)),
        "qvel": np.zeros((n_steps, model.nv)),
        "ctrl": np.zeros((n_steps, model.nu)),
    }

    mujoco.mj_resetData(model, data)

    for i in range(n_steps):
        t = i * dt
        # Sinusoidal control for interesting motion
        data.ctrl[0] = 5.0 * np.sin(2.0 * t)
        data.ctrl[1] = 3.0 * np.cos(3.0 * t)

        # Record state
        trajectory["time"][i] = t
        trajectory["qpos"][i] = data.qpos.copy()
        trajectory["qvel"][i] = data.qvel.copy()
        trajectory["ctrl"][i] = data.ctrl.copy()

        mujoco.mj_step(model, data)

    return trajectory


def save_trajectory(trajectory, filename="trajectory.npz"):
    """Save trajectory to compressed NumPy file."""
    np.savez_compressed(filename, **trajectory)
    print(f"Saved trajectory to {filename} ({trajectory['time'].shape[0]} steps)")


def load_trajectory(filename="trajectory.npz"):
    """Load trajectory from file."""
    data = np.load(filename)
    trajectory = {key: data[key] for key in data.files}
    print(f"Loaded trajectory: {trajectory['time'].shape[0]} steps, "
          f"{trajectory['time'][-1]:.1f}s duration")
    return trajectory


def playback_trajectory(model, trajectory):
    """Replay a recorded trajectory in the viewer."""
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    n_steps = len(trajectory["time"])

    print(f"Playing back {n_steps} steps...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(n_steps):
            if not viewer.is_running():
                break

            # Set state from recorded trajectory
            data.qpos[:] = trajectory["qpos"][i]
            data.qvel[:] = trajectory["qvel"][i]
            data.ctrl[:] = trajectory["ctrl"][i]

            # Forward kinematics (no stepping — just visualize)
            mujoco.mj_forward(model, data)
            viewer.sync()

    print("Playback complete.")


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Step 1: Record
    print("=== Recording trajectory ===")
    traj = record_trajectory(model, data, duration=5.0)

    # Step 2: Save
    save_trajectory(traj, "/tmp/demo_trajectory.npz")

    # Step 3: Load (demonstrates persistence)
    loaded_traj = load_trajectory("/tmp/demo_trajectory.npz")

    # Step 4: Playback
    print("\n=== Playing back recorded trajectory ===")
    playback_trajectory(model, loaded_traj)


if __name__ == "__main__":
    main()
