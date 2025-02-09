"""
AIAgent.py
"""

import torch
from ModelL import ModelL
from ReplayMemory import ReplayMemory
from EpsilonAlgo import EpsilonAlgo
from AITrainer import AITrainer

class AIAgent:
    def __init__(self, ini, log, stats):
        self.ini = ini
        self.log = log
        self.stats = stats
        self.model = ModelL(ini, log, stats)
        self.epsilon_algo = EpsilonAlgo(ini, log, stats)
        self.memory = ReplayMemory(ini)
        self.trainer = AITrainer(ini, log, stats, self.model)
        self.log.log('AIAgent initialization:     [OK]')
        self.last_dirs = [ 0, 0, 1, 0 ]

    def get_move(self, state):
        random_move = self.epsilon_algo.get_move()
        if random_move:
            return random_move # Random move was returned
        # AI agent based action
        final_move = [0, 0, 0]
        #state0 = torch.tensor(state, dtype=torch.float) # Convert to a tensor
        prediction = self.model(state) # Get the prediction
        move = torch.argmax(prediction).item() # Get the move
        final_move[move] = 1 # Set the move
        return final_move # Return
    
    def played_game(self, score):
        self.epsilon_algo.played_game()
 
    def remember(self, state, action, reward, next_state, done):
        # Store the state, action, reward, next_state, and done in memory
        # Recall that memory is a deque, so it will automatically remove the oldest memory 
        # if the memory exceeds MAX_MEMORY
        self.memory.append((state, action, reward, next_state, done))

    def reset_epsilon_injected(self):
        self.epsilon_algo.reset_injected()

    def train_long_memory(self):
        # Get the states, actions, rewards, next_states, and dones from the mini_sample
        mini_sample = self.memory.get_memory()
        states, actions, rewards, next_states, dones = zip(*mini_sample)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
