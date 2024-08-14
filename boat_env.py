import sys

import gymnasium as gym

import gym_sea_env

env = gym.make('sea_boat_env/SeaBoatEnv-v1', render_mode="human", verbose=True, continuous=True)
observation, info = env.reset()

for _ in range(10000):
  observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
  if terminated or truncated:
    observation, info = env.reset()

env.close()
