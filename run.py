import gymnasium
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.env_checker import check_env

import gym_sea_env
from gym_sea_env.wrappers import RelativePosition

env = gymnasium.make('gym_sea_env/SeaBoatEnv-v1', render_mode="human", verbose=True, continuous=True)

# check_env(env)

observation, info = env.reset()

for _ in range(10000):
  observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
  if terminated or truncated:
    observation, info = env.reset()

env.close()



# Using Wrappers
# wrapped_env = RelativePosition(env)
# print(wrapped_env.reset()) # E.g.  [-3  3], {}
#
# wrapped_env = FlattenObservation(env)
# print(wrapped_env.reset()) # E.g.  [3 0 3 3], {}
