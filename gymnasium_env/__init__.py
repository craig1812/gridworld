from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v20",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)