"""Using MuJoCo with Gymnasium — quickstart example."""

import gymnasium as gym
import numpy as np


def main():
    # Gymnasium ships with built-in MuJoCo environments
    env = gym.make("InvertedPendulum-v5", render_mode="human")

    obs, info = env.reset(seed=42)
    total_reward = 0.0

    for step in range(1000):
        # Random policy (replace with your controller/RL agent)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {step}, reward: {total_reward:.1f}")
            obs, info = env.reset()
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
