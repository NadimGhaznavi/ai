"""
snake/Agent.py

    Database 4 Everything
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai
    License: GPL 3.0
"""

import torch
from ModelL import ModelL
from EpsilonAlgo import EpsilonAlgo
from ReplayMemory import ReplayMemory
from AITrainer import AITrainer
from ModelL import ModelL


class AIAgent:

    def __init__(self):
        self.model = ModelL()
        self.epsilon_algo = EpsilonAlgo()
        self.memory = ReplayMemory()
        self.model = ModelL()
        self.trainer = AITrainer(self.model)

    def get_model(self):
        return self.model

    def get_move(self, state):
        random_move = self.epsilon_algo.get_move()  # Explore with epsilon
        if random_move != False:
            return random_move  # Random move was returned

        # Exploit with an AI agent based action
        final_move = [0, 0, 0]
        if type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.float)  # Convert to a tensor
        prediction = self.model(state)  # Get the prediction
        move = torch.argmax(prediction).item()  # Select the move with the highest value
        final_move[move - 1] = 1  # Set the move
        return final_move  # Return

    def get_optimizer(self):
        return self.trainer.get_optimizer()

    def played_game(self, score):
        self.epsilon_algo.played_game()

    def remember(self, state, action, reward, next_state, done):
        # Store the state, action, reward, next_state, and done in memory
        self.memory.append((state, action, reward, next_state, done))

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.trainer.set_optimizer(optimizer)

    def train_long_memory(self):
        # Get the states, actions, rewards, next_states, and dones from the mini_sample
        memory = self.memory.get_memory()

        for state, action, reward, next_state, done in memory:
            self.trainer.train_step(state, action, reward, next_state, [done])

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, [done])
