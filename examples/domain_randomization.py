"""Domain Randomization example — vary physics parameters across episodes."""

import numpy as np
import mujoco
import mediapy

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="3 3 0.1" rgba=".8 .8 .8 1"/>
    <body name="ball" pos="0 0 2">
      <joint type="free"/>
      <geom name="ball_geom" type="sphere" size="0.15" mass="1" rgba="0 0.5 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def simulate_episode(gravity_z: float, mass: float, restitution: float, duration: float = 2.0):
    """Run one episode with randomized parameters."""
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Randomize physics
    model.opt.gravity[2] = gravity_z
    model.body_mass[1] = mass  # body 0 is world

    # Set bounce (solref controls restitution)
    # Lower solref[0] = stiffer contact = more bounce
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    model.geom_solref[geom_id, 0] = 0.01 * (1 - restitution)

    # Give horizontal velocity
    data.qvel[0] = 2.0

    renderer = mujoco.Renderer(model, height=240, width=320)
    frames = []
    fps = 30

    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * fps:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

    renderer.close()
    return frames


def main():
    rng = np.random.default_rng(42)
    all_frames = []

    print("Running 4 episodes with randomized physics...")
    for i in range(4):
        gravity_z = rng.uniform(-15.0, -5.0)
        mass = rng.uniform(0.5, 3.0)
        restitution = rng.uniform(0.3, 0.95)

        print(f"  Episode {i+1}: gravity={gravity_z:.1f}, mass={mass:.2f}, restitution={restitution:.2f}")
        frames = simulate_episode(gravity_z, mass, restitution)
        all_frames.append(frames)

    # Save side-by-side comparison
    min_len = min(len(f) for f in all_frames)
    combined = []
    for t in range(min_len):
        row1 = np.concatenate([all_frames[0][t], all_frames[1][t]], axis=1)
        row2 = np.concatenate([all_frames[2][t], all_frames[3][t]], axis=1)
        combined.append(np.concatenate([row1, row2], axis=0))

    mediapy.write_video("domain_randomization.mp4", combined, fps=30)
    print(f"Saved {len(combined)} frames to domain_randomization.mp4")


if __name__ == "__main__":
    main()
