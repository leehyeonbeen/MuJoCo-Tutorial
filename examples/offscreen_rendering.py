"""Offscreen rendering and video recording with MuJoCo (works in notebooks/headless)."""

import mujoco
import numpy as np
import mediapy

XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
    <body pos="0 0 2">
      <joint type="free"/>
      <geom type="sphere" size="0.2" rgba="0 0.5 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    frames = []
    duration = 3.0  # seconds
    fps = 30

    while data.time < duration:
        mujoco.mj_step(model, data)

        if len(frames) < data.time * fps:
            renderer.update_scene(data)
            frames.append(renderer.render().copy())

    renderer.close()

    # Save as video
    mediapy.write_video("bouncing_ball.mp4", frames, fps=fps)
    print(f"Saved {len(frames)} frames to bouncing_ball.mp4")

    # In a notebook, you can display inline:
    # mediapy.show_video(frames, fps=fps)


if __name__ == "__main__":
    main()
