import mujoco
import numpy as np
import mediapy as media

model=mujoco.MjModel.from_xml_path("HYEONBEEN/xml/011_named_access.xml")
model.ngeom # num geom
model.geom_rgba # rgba allocated to geoms, top to bottom

# access model attr by name
model.geom("green_sphere").rgba
model.geom("green_sphere").id # id starts from 0, from top to bottom
model.geom(model.geom("green_sphere").id)

data=mujoco.MjData(model)
print(data.geom_xpos) # derived geometric -> computed after calling mj_kinematics or mj_forward or mj_step
mujoco.mj_kinematics(model, data)
print(data.geom_xpos)
pass