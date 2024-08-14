from typing import Any, Dict

import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

import gym_sea_env

# Create environment
env = gym.make("gym_sea_env/SeaBoatEnv-v1", render_mode="rgb_array")

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
try: model = PPO.load("sea_boat_model", env=env, print_system_info=True)
except Exception as e: model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log="./ppo_seaboat_tensorboard")


# Train the agent and display a progress bar
model.learn(total_timesteps=int(7e3), progress_bar=True)
# model.learn(total_timesteps=int(2e5), progress_bar=True)

# Save the agent
model.save("sea_boat_model")

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

# # Parallel environments
# vec_env = make_vec_env("sea_world_proto/SeaBoatEnv-v1", n_envs=5)

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=25000)

# # model = PPO.load("ppo_seaboat")

# obs = vec_env.reset()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")

# model.save("ppo_seaboat")


# # Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=4)

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")



# env = gym.make("sea_world_proto/SeaBoatEnv-v1", render_mode="rgb_array")

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(10):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render("human")


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True

# Video
# model = PPO("MlpPolicy", "CartPole-v1", tensorboard_log="runs/", verbose=1)
# video_recorder = VideoRecorderCallback(gym.make("CartPole-v1"), render_freq=5000)
# model.learn(total_timesteps=int(5e4), callback=video_recorder)