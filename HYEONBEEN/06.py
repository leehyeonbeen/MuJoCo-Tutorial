import mujoco
import numpy as np
import mediapy as media
import time


def make_arm():
    model = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/061_arm.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=400, width=500)
    renderer.update_scene(data)
    im = renderer.render().copy()
    media.write_image("HYEONBEEN/vid/061.png", im)


def composite():
    model = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/063_composite.xml")
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            while data.time < 3:
                mujoco.mj_step(model, data)


def mjspec():
    # MjSpec supports programmatic modeling.
    # Example: programmatic model editing (MuJoCo 3.x spec API)
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(pos=[0, 0, 2])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])

    body = spec.worldbody.add_body(name="dynamic_box", pos=[0, 0, 1])
    body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.1, 0.1, 0.1],
        mass=1.0,
        rgba=[0, 0.7, 0, 1],
    )

    model_spec = spec.compile()
    data_spec = mujoco.MjData(model_spec)
    mujoco.mj_forward(model_spec, data_spec)

    renderer_s = mujoco.Renderer(model_spec, height=300, width=400)
    renderer_s.update_scene(data_spec)
    media.show_image(renderer_s.render())
    renderer_s.close()
    print("Model created programmatically with MjSpec!")


def equality():
    model = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/064_equality_const.xml")
    data = mujoco.MjData(model)

    times, master_angles, follower_angles = [], [], []
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[0] = 1.0 * np.sin(data.time)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


# make_arm()
# composite()
equality()
