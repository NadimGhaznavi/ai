"""
AIAgent.py
"""

import torch
import time
from ModelL import ModelL
from ModelRNN import ModelRNN
from ModelRNNT import ModelRNNT
from ModelRNNX import ModelRNNX
from ModelT import ModelT
from ModelCNN import ModelCNN
from ReplayMemory import ReplayMemory
from EpsilonAlgo import EpsilonAlgo
from AITrainer import AITrainer
from NuAlgo import NuAlgo

class AIAgent:
    def __init__(self, ini, log, stats):
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        if ini.get('model') == 'linear':
            self.model = ModelL(ini, log, stats)
        elif ini.get('model') == 'rnn':
            self.model = ModelRNN(ini, log, stats)
        elif ini.get('model') == 't':
            self.model = ModelT(ini, log, stats)
        elif ini.get('model') == 'rnnt':
            self.model = ModelRNNT(ini, log, stats)
        elif ini.get('model') == 'rnnx':
            self.model = ModelRNNX(ini, log, stats)
        elif ini.get('model') == 'cnn':
            self.model = ModelCNN(ini, log, stats)
        else:
            raise Exception(f"Unknown model type {ini.get('model')}")
        self.epsilon_algo = EpsilonAlgo(ini, log, stats)
        self.nu_algo = NuAlgo(ini, log, stats)
        self.memory = ReplayMemory(ini)
        self.trainer = AITrainer(ini, log, stats, self.model)
        self.log.log('AIAgent initialization:     [OK]')
        self.last_dirs = [ 0, 0, 1, 0 ]

    def cleanup(self):
        self.stats.save()
        self.ini.save()

    def get_model(self):
        return self.model

    def get_move(self, state):
        random_move = self.epsilon_algo.get_move()
        if random_move:
            return random_move # Random move was returned
        random_move = self.nu_algo.get_move(self.stats.get('game', 'score'))
        if random_move:
            return random_move # Random move was returned
        

        # AI agent based action
        final_move = [0, 0, 0]
        if type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.float) # Convert to a tensor
        prediction = self.model(state) # Get the prediction
        move = torch.argmax(prediction).item() # Get the move
        final_move[move] = 1 # Set the move
        return final_move # Return
        
    def played_game(self, score):
        self.epsilon_algo.played_game()
        self.nu_algo.played_game(score)
        self.trainer.reset_steps()
        self.model.reset_steps()
        self.stats.set('agent', 'score', score)
 
    def remember(self, state, action, reward, next_state, done):
        # Store the state, action, reward, next_state, and done in memory
        # Recall that memory is a deque, so it will automatically remove the oldest memory 
        # if the memory exceeds MAX_MEMORY
        self.memory.append((state, action, reward, next_state, done))

    def reset_epsilon_injected(self):
        self.epsilon_algo.reset_injected()

    def reset_nu_injected(self):
        self.nu_algo.reset_injected()

    def train_long_memory(self):
        # Get the states, actions, rewards, next_states, and dones from the mini_sample
        if self.ini.get('model') != 'cnn':
            mini_sample = self.memory.get_memory()
            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
