from gymnasium.envs.registration import register

register(
     id="gym_sea_env/SeaBoatEnv-v1",
     entry_point="gym_sea_env.envs:SeaBoatEnv",
     max_episode_steps=4096,
)
