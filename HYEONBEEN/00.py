import mujoco
import numpy as np
import mediapy as media
import matplotlib
import time

print(mujoco.__version__)

# A free-falling ball
# model = mujoco.MjModel.from_xml_string("HYEONBEEN/xml/001_world_description.xml")
# data = mujoco.MjData(model)
# renderer = mujoco.Renderer(model, height=400, width=500)
# frames = []
# while data.time < 3:
#     mujoco.mj_step(model, data)
#     if len(frames) < data.time * 30:
#         renderer.update_scene(data)
#         frames.append(renderer.render().copy())
# renderer.close()


# Single pendulum
# m = mujoco.MjModel.from_xml_string("HYEONBEEN/xml/002_pendulum.xml")
# d = mujoco.MjData(m)
# # IC
# d.qpos[0] = np.pi / 2  # joint dof = generalized coordinates q
# renderer = mujoco.Renderer(model, height=400, width=500)
# frames = []
# fps = 60
# steps_per_frame = round(1 / fps) / model.opt.timestep
# with mujoco.viewer.launch_passive(m, d) as viewer:
#     viewer.cam.distance = 3
#     viewer.cam.lookat = np.ones(3) * 10
#     while d.time < 5:

#         t0 = time.perf_counter()
#         mujoco.mj_step(m, d)
#         viewer.sync()

#         leftover = 1 / fps - (time.perf_counter() - t0)
#         if leftover > 0:
#             time.sleep(leftover)

#         if len(frames) < d.time * 30:
#             renderer.update_scene(d)
#             frames.append(renderer.render().copy())

# PD control of angular position
# model = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/003_control.xml")
# data = mujoco.MjData(model)
# target = np.pi / 4  # target angle to reach
# renderer = mujoco.Renderer(model, height=400, width=400)
# frames = []
# while data.time < 6.0:
#     current_angle = data.qpos[0]
#     current_speed = data.qvel[0]
#     error = target - current_angle
#     motor_command = 10.0 * error - 3.0 * current_speed  # PD controller
#     data.ctrl[0] = np.clip(motor_command, -5, 5)  # joint torque limit -5, 5
#     mujoco.mj_step(model, data)

#     if len(frames) < data.time * 30:  # stack until time * fps
#         renderer.update_scene(data)
#         frames.append(renderer.render().copy())
# renderer.close()
# media.write_video("HYEONBEEN/vid/003_control.mp4", frames, fps=30)

pass
