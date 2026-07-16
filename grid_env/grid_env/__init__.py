from gymnasium.envs.registration import register

if __name__ == "grid_env":
    register(
        id="grid_env/GridWorld-v0",
        entry_point="grid_env.envs:GridWorldEnv",
    )
