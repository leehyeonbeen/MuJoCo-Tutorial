"""MJX (JAX-accelerated MuJoCo) — batched parallel simulation example.

Requires: pip install mujoco-mjx jax
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    import mujoco
except ImportError:
    print("This example requires JAX and MJX:")
    print("  pip install mujoco-mjx jax[cuda12]  # for GPU")
    print("  pip install mujoco-mjx jax[cpu]      # for CPU only")
    exit(1)

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="ball" pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    # Create MuJoCo model
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    # Convert to MJX (JAX-compatible) model
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)

    # JIT-compile the step function
    jit_step = jax.jit(mjx.step)

    # Batch simulation: simulate 1000 environments in parallel
    batch_size = 1000

    # Vectorize over different initial velocities
    rng = jax.random.PRNGKey(0)
    random_velocities = jax.random.normal(rng, (batch_size, 6)) * jnp.array([2, 2, 5, 1, 1, 1])

    # Create batched initial data
    batched_data = jax.vmap(lambda vel: mjx_data.replace(qvel=vel))(random_velocities)

    # Step all environments in parallel
    @jax.jit
    def simulate_batch(data, n_steps=100):
        def step_fn(data, _):
            data = mjx.step(mjx_model, data)
            return data, data.qpos[:3]  # track ball position

        final_data, trajectories = jax.lax.scan(step_fn, data, None, length=n_steps)
        return final_data, trajectories

    # Run batched simulation
    print(f"Simulating {batch_size} environments in parallel...")
    batched_step = jax.vmap(simulate_batch)
    final_data, trajectories = batched_step(batched_data)

    # Results
    final_positions = np.array(final_data.qpos[:, :3])
    print(f"Final positions (mean): {final_positions.mean(axis=0)}")
    print(f"Final positions (std):  {final_positions.std(axis=0)}")
    print(f"Trajectory shape: {trajectories.shape}  (batch, timesteps, xyz)")


if __name__ == "__main__":
    main()
