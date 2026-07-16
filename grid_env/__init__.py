"""GridWorld registration for the tabularRL package."""

from gymnasium.envs.registration import register

register(
    id="grid_env/GridWorld-v0",
    entry_point="tabularRL.grid_env.grid_env.envs:GridWorldEnv",
)
