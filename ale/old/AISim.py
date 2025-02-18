"""
AISim.py

The code that runs a simulation. Where a simulation consists of
a game environment.

"""
import gymnasium as gym
from SimConfig import SimConfig


class AISim:
    def __init__(self, rom, agent=None):
        self.rom = rom
        self.env = gym.make(rom, render_mode="human")

        self.agent = agent
        self.config = SimConfig(rom)
        self.num = 0 # Number of simulations run

        if agent is None:
            # Setup a dummy agent if none was passed in
            self.agent = AIAgent(self.env)

        # Extract ROM specific metrics from the env
        print(self.env.action_space)

        # Reset the environment
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.observation = False
        self.info = False
        
        # Reset the environment with a known seed, doesn't work
        self.observation, self.info = self.env.reset(seed=1970)

    def reset(self):
        self.observation, self.info = self.env.reset()

    def run(self):
        num = 0
        print(f"Starting {self.rom}, simulation #{self.config.get('sim_num')}")
        while num < self.config.get('max_epochs'):
            num += 1
            # Run a single epoch
            print(f"Starting epoch: {num}", end='')
            game_over = False # Initialize to False
            num_moves = 0 # Number of moves in the game
            while not game_over:
                # New game!!
                num_moves += 1   # Increment the number of moves
                # Game is not over...
                # Select an action from the agent
                action = self.agent.get_action()
                # Execute the action
                self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
                # Set the episode over flag if the result of the step was termination or truncation
                game_over = self.terminated or self.truncated
            # Incrment our own counter
            print(f" - {num_moves} moves")
            self.config.set('epoch_' + str(num), 'moves# ' + str(num_moves))
            self.config.incr('epochs')
            # Reset the environment
            self.reset()

        self.env.close()

    
class AIAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self):
        return self.env.action_space.sample()

