"""Generate GIF previews of examples for the README."""

import os
import sys
import numpy as np

os.environ["MUJOCO_GL"] = "egl"

import mujoco
import mediapy as media
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
OUTPUT_DIR = Path(__file__).parent.parent / "img" / "previews"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECORDINGS = {
    "trajectory_tracking.py": {"duration": 6.0, "fps": 30},
    "lqr_control.py": {"duration": 5.0, "fps": 30},
    "pick_and_place.py": {"duration": 10.0, "fps": 20},
    "impedance_control.py": {"duration": 5.0, "fps": 30},
}


def record_example(name, duration=5.0, fps=30):
    """Record an example to GIF by running its simulation headlessly."""
    import importlib.util

    path = EXAMPLES_DIR / name
    if not path.exists():
        print(f"  Skipping {name} - not found")
        return

    spec = importlib.util.spec_from_file_location("example", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "XML"):
        print(f"  Skipping {name} - no XML constant")
        return

    model = mujoco.MjModel.from_xml_string(mod.XML)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=400, width=600)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -20
    camera.distance = 2.0
    camera.lookat[:] = [0, 0, 0.5]

    n_frames = int(duration * fps)
    steps_per_frame = int(1.0 / (model.opt.timestep * fps))
    frames = []

    # Set up controller
    controller = None
    if hasattr(mod, "HopperFSM"):
        controller = mod.HopperFSM()
    elif hasattr(mod, "PickAndPlaceController"):
        controller = mod.PickAndPlaceController()

    # LQR init
    lqr_K = None
    if hasattr(mod, "compute_lqr_gain") and hasattr(mod, "linearize"):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        A, B = mod.linearize(model, data)
        Q = np.diag([1.0, 10.0, 0.1, 1.0])
        R = np.array([[0.01]])
        lqr_K = mod.compute_lqr_gain(A, B, Q, R)
        data.qpos[:] = [0.0, 0.1]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    for frame_idx in range(n_frames):
        for _ in range(steps_per_frame):
            if controller and hasattr(controller, "get_action"):
                data.ctrl[:] = controller.get_action(data, model.opt.timestep)
            elif controller and hasattr(controller, "get_ctrl"):
                data.ctrl[:] = controller.get_ctrl(model.opt.timestep)
            elif lqr_K is not None:
                x = np.concatenate([data.qpos, data.qvel])
                u = -lqr_K @ x
                data.ctrl[0] = np.clip(u[0], -20, 20)
            elif hasattr(mod, "circle_trajectory"):
                ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
                jacp = np.zeros((3, model.nv))
                jacr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, jacp, jacr, ee_id)
                pos_des, vel_des = mod.circle_trajectory(data.time)
                pos_curr = data.site_xpos[ee_id]
                vel_curr = jacp @ data.qvel
                force = 200 * (pos_des - pos_curr) + 40 * (vel_des - vel_curr)
                data.ctrl[:] = jacp.T @ force
            mujoco.mj_step(model, data)

        renderer.update_scene(data, camera)
        frames.append(renderer.render().copy())

    renderer.close()

    output_path = OUTPUT_DIR / f"{Path(name).stem}.gif"
    media.write_video(str(output_path), frames, fps=fps, codec="gif")
    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path.name} ({len(frames)} frames, {size_kb:.0f} KB)")


def main():
    print("Recording example GIFs...")
    for name, settings in RECORDINGS.items():
        print(f"\n  Recording {name}...")
        try:
            record_example(name, **settings)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! GIFs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
