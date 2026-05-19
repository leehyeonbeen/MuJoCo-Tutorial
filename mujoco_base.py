"""Modern MuJoCo simulation base class using mujoco >= 3.0 viewer."""

import mujoco
import mujoco.viewer
import numpy as np


class MuJoCoBase:
    """Base class for MuJoCo simulations with the modern Python viewer.

    Subclass this and implement `reset()` and `controller()`.
    Optionally override `on_step()` for per-step logic.

    Usage:
        class MySimulation(MuJoCoBase):
            def reset(self):
                # Set initial state
                pass

            def controller(self, model, data):
                # Set data.ctrl[:] here
                pass

        sim = MySimulation("my_model.xml")
        sim.run()
    """

    def __init__(self, xml_path: str, dt: float = 0.002):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)

    def reset(self):
        """Initialize/reset the simulation state. Override in subclass."""
        mujoco.mj_resetData(self.model, self.data)

    def controller(self, model, data):
        """Compute control signals. Override in subclass. Set data.ctrl[:] here."""
        pass

    def on_step(self, model, data):
        """Called after each simulation step. Override for custom logic."""
        pass

    def run(self, duration: float = 10.0):
        """Run interactive simulation with the built-in MuJoCo viewer."""
        self.reset()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and self.data.time < duration:
                self.controller(self.model, self.data)
                mujoco.mj_step(self.model, self.data)
                self.on_step(self.model, self.data)
                viewer.sync()

    def run_headless(self, duration: float = 10.0) -> list[np.ndarray]:
        """Run simulation headless and return rendered frames."""
        self.reset()
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        frames = []

        while self.data.time < duration:
            self.controller(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
            self.on_step(self.model, self.data)

            # Render at ~30fps
            if int(self.data.time / (1 / 30)) > len(frames):
                renderer.update_scene(self.data)
                frames.append(renderer.render().copy())

        renderer.close()
        return frames
