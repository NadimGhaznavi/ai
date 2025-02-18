import gymnasium as gym
from AISim import AISim

# Create the environment
env = gym.make('Breakout-v3', render_mode='human')

sim = AISim(env)
sim.run()
