import matplotlib

matplotlib.use("agg")
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media

m = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/041_control.xml")
d = mujoco.MjData(m)

target_angle = np.pi / 3  # 60 degrees
kp, kd = 10.0, 2.0

times, angles, controls = [], [], []
mujoco.mj_resetData(m, d)

while d.time < 5.0:
    angle = d.qpos[0]
    velocity = d.qvel[0]
    tau = kp * (target_angle - angle) - kd * velocity  # -> actuator torque
    d.ctrl[0] = np.clip(tau, -5, 5)

    mujoco.mj_step(m, d)
    times.append(d.time)
    angles.append(angle)
    controls.append(d.ctrl[0])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(times, np.degrees(angles), label="Angle")
ax1.axhline(np.degrees(target_angle), color="r", linestyle="--", label="Target")
ax1.set_ylabel("Angle (deg)")
ax1.legend()
ax1.grid(True)
ax2.plot(times, controls, color="orange")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Torque (Nm)")
ax2.grid(True)
plt.suptitle("PD Control of Pendulum")
plt.tight_layout()
plt.show()
fig.savefig("HYEONBEEN/vid/041_control.png", dpi=300)


plt.figure(figsize=(10, 4))
for kp, kd, label in [
    (5, 1, "Underdamped"),
    (10, 6, "Critically damped"),
    (10, 15, "Overdamped"),
]:
    mujoco.mj_resetData(m, d)
    times, angles = [], []
    while d.time < 5.0:
        d.ctrl[0] = np.clip(kp * (target_angle - d.qpos[0]) - kd * d.qvel[0], -5, 5)
        mujoco.mj_step(m, d)
        times.append(d.time)
        angles.append(d.qpos[0])
    plt.plot(times, np.degrees(angles), label=label)

plt.axhline(np.degrees(target_angle), color="k", linestyle="--", alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.title("Effect of PD Gains")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("HYEONBEEN/vid/041_control_gaintuning.png")

# Servo model (position control)
m = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/042_control_posctrl.xml")
d = mujoco.MjData(m)
t, a = [], []
while d.time < 5:
    d.ctrl[0] = np.pi / 4 if d.time < 2.5 else -np.pi / 4 # desired position
    mujoco.mj_step(m, d)
    t.append(d.time)
    a.append(d.qpos[0])
plt.figure(figsize=(10, 4))
plt.plot(t, np.degrees(a))
plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.title("Position Servo (built-in PD)")
plt.savefig("HYEONBEEN/vid/042_posctrl.png", dpi=300)

# Servo model (velocity control)
m = mujoco.MjModel.from_xml_path("HYEONBEEN/xml/042_control_velctrl.xml")
d = mujoco.MjData(m)
t, p,v = [], [],[]
while d.time < 5:
    d.ctrl[0] = 2 if d.time < 2.5 else -2 # desired position
    mujoco.mj_step(m, d)
    t.append(d.time)
    p.append(d.qpos[0])
    v.append(d.qvel[0])
fig,ax=plt.subplots(2,1)
ax[0].plot(t,np.rad2deg(p))
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Angle (deg)")
ax[0].set_title("Position Servo (built-in PD)")
ax[1].plot(t,v)
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Vel (rad/s)")
ax[1].set_title("Velocity Servo (built-in PD)")
fig.tight_layout()
fig.savefig("HYEONBEEN/vid/042_velctrl.png", dpi=300)

# Using keyframes
model=mujoco.MjModel.from_xml_path("HYEONBEEN/xml/042_control_keyframe.xml")
data=mujoco.MjData(model)
renderer=mujoco.Renderer(model)
images=[]
for i in range(model.nkey):
    key_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
    mujoco.mj_resetDataKeyframe(model, data, i)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    im=renderer.render().copy()
    media.write_image(f"HYEONBEEN/vid/043_keyframe_{i}.png",im)
renderer.close()
    