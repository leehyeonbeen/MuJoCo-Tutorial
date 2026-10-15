import mujoco
import numpy as np
import mediapy as media

# Basic
model = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/021_rendering.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data)
media.write_image("HYEONBEEN/vid/021_render.png", renderer.render())

# Free camera
camera = mujoco.MjvCamera()
camera.azimuth = 135  # planar rotation, 90 by default (Look along +y direction, azimuth=0 means +x direction)
camera.elevation = -20  # spherical up/down, -45 by default
camera.distance = 2.0  # 2 by default
camera.lookat[:] = [0, 0, 0.2]  # camera origin, O by default

renderer.update_scene(data, camera=camera)
media.write_image("HYEONBEEN/vid/021_render_cam.png", renderer.render())

images = []
for azimuth in [0, 90, 180, 270]:
    camera.azimuth = azimuth
    renderer.update_scene(data, camera=camera)
    im = renderer.render().copy()
    images.append(im)
    media.write_image(f"HYEONBEEN/vid/021_render_azim{azimuth}.png", im)


# Body-fixed camera (in xml)
model2 = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/021_rendering_cam.xml")
data2 = mujoco.MjData(model2)
mujoco.mj_forward(model2, data2)
renderer2 = mujoco.Renderer(model2, height=300, width=400)
images = []
for cam_name in ["top_down", "side"]:
    renderer2.update_scene(data2, camera=cam_name)
    im = renderer2.render().copy()
    images.append(im)
    media.write_image(f"HYEONBEEN/vid/022_render_{cam_name}.png", im)


# Depth and segmentation rendering
model4 = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/021_rendering_cam.xml")
data4 = mujoco.MjData(model)
mujoco.mj_forward(model4, data4)
renderer4 = mujoco.Renderer(model4, height=480, width=640)

# RGB
renderer4.update_scene(data4)
rgb = renderer4.render().copy()  # (H,W,3)
# Depth
renderer4.enable_depth_rendering()
renderer4.update_scene(data4)
depth = renderer4.render().copy()  # (H,W)
renderer4.disable_depth_rendering()
# Segmentation
renderer4.enable_segmentation_rendering()
renderer4.update_scene(data4)
seg = renderer4.render().copy()  # (H,W,2)
renderer4.disable_segmentation_rendering()

renderer4.close()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

depth=np.clip(depth,0, 3) # too far -> too high depth -> invisible

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(rgb)
axes[0].set_title("RGB")
axes[1].imshow(depth, cmap="viridis")
axes[1].set_title("Depth")
axes[2].imshow(seg[..., 0], cmap="tab10")
axes[2].set_title("Segmentation 1 (obj id)") # background + every geom 5 uniques
axes[3].imshow(seg[..., 1], cmap="tab10")
axes[3].set_title("Segmentation 2 (obj type)") # background, geom 2 uniques
for ax in axes:
    ax.axis("off")
plt.tight_layout()
fig.savefig("HYEONBEEN/vid/023_render_depth_segmentation.png", dpi=300)

pass
