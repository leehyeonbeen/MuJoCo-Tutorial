# Examples

Standalone scripts demonstrating MuJoCo concepts. Each is self-contained with inline MJCF.

Run any example: `python examples/<name>.py`

## Basics

| Example | Concept | Difficulty |
|---------|---------|:----------:|
| `projectile.py` | External forces (drag), `xfrc_applied` | ⭐ |
| `pendulum_control.py` | PD control, `data.ctrl` | ⭐ |
| `double_pendulum.py` | Multi-joint control, energy shaping | ⭐⭐ |
| `offscreen_rendering.py` | `mujoco.Renderer`, video export | ⭐ |
| `record_playback.py` | Save/load trajectories with NumPy | ⭐ |

## Control Theory

| Example | Concept | Difficulty |
|---------|---------|:----------:|
| `trajectory_tracking.py` | Task-space PD, Jacobian transpose | ⭐⭐ |
| `operational_space_control.py` | Dynamics compensation, Λ matrix | ⭐⭐⭐ |
| `lqr_control.py` | Linearization, Riccati equation, LQR | ⭐⭐⭐ |
| `impedance_control.py` | Compliance, spring-damper in task space | ⭐⭐ |
| `inverse_kinematics.py` | Damped least-squares IK | ⭐⭐ |

## Robotics

| Example | Concept | Difficulty |
|---------|---------|:----------:|
| `locomotion_controller.py` | Finite state machine, hopper gait | ⭐⭐ |
| `pick_and_place.py` | Gripper, waypoint sequencing | ⭐⭐ |
| `hopper.py` | Open-loop jumping | ⭐⭐ |
| `menagerie_robot.py` | Loading Menagerie models | ⭐ |

## Advanced

| Example | Concept | Difficulty |
|---------|---------|:----------:|
| `contact_forces.py` | Contact detection, force reading | ⭐⭐ |
| `domain_randomization.py` | Parameter variation, sim-to-real | ⭐⭐ |
| `mjx_parallel.py` | GPU-parallel simulation (JAX) | ⭐⭐⭐ |
| `gymnasium_env.py` | RL environments | ⭐⭐ |
