import matplotlib

matplotlib.use("agg")
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import time


def ex01():
    model = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/051_contact.xml")
    data = mujoco.MjData(model)
    contact_times, contact_forces, heights = [], [], []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 2:
            mujoco.mj_step(model, data)
            heights.append(data.qpos[2])

            total_force = 0.0
            for i in range(data.ncon):
                force = np.zeros(6)  # container array
                mujoco.mj_contactForce(
                    model, data, i, force
                )  # pass the container to the function
                total_force += np.linalg.norm(force[:3])
            contact_times.append(data.time)
            contact_forces.append(total_force)
            viewer.sync()
            time.sleep(model.opt.timestep)
    # while viewer.is_running():
    #     viewer.sync()
    #     time.sleep(1)


def ex02():
    model = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/052_contact_params.xml")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.distance = 5
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.zeros(3)
        for friction in [0.1, 0.5, 1, 2]:
            model.geom("ramp").friction[0] = friction
            model.geom("box").friction[0] = friction

            positions = []
            while data.time < 2:
                mujoco.mj_step(model, data)
                positions.append(data.qpos[0])  # x position
                viewer.sync()
                time.sleep(model.opt.timestep)
            plt.plot(
                np.linspace(0, 2, len(positions)),
                positions,
                label=f"'friction={friction}",
            )
            mujoco.mj_resetData(model, data)
        plt.xlabel("Time (s)")
        plt.ylabel("X position (m)")
        plt.title("Effect of friction on sliding")
        plt.legend()
        plt.savefig("HYEONBEEN/vid/051_contact_params.png", dpi=300)


def ex03():
    model = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/053_contact_filtering.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=300, width=400)
    frames = []
    while data.time < 2:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * 30:
            renderer.update_scene(data)
            im = renderer.render().copy()
            frames.append(im)
    media.write_video("HYEONBEEN/vid/052_contact_filtering.mp4", frames)


ex01()
ex02()
ex03()
