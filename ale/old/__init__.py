from gymnasium.envs.registration import register
from gymnasium_env.envs.grid_world import GridWorldEnv

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)