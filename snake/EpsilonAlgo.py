"""
snake/EpsilonAlgo.py

    Database 4 Everything
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai
    License: GPL 3.0


A class to encapsulate the functionality of the epsilon algorithm. The algorithm
injects random moves at the beginning of the simulation. The amount of moves
is controlled by the epsilon_value parameter which is in the AISnakeGame.ini and
can also be passed in when invoking the main asg.py front end.
"""

import random
from random import randint
import os, sys


class EpsilonAlgo:
    def __init__(self):
        # Set this random seed so things are repeatable
        random.seed(1970)
        self.epsilon_value = 0.6
        self.epsilon_min = 0
        self.epsilon_decay = 0.9
        self.epsilon = self.epsilon_value
        self.num_games = 0
        self.injected = 0
        self.depleted = False

    def get_move(self):
        if random.random() < self.epsilon:
            rand_move = [0, 0, 0]
            rand_idx = randint(0, 2)
            rand_move[rand_idx] = 1
            self.injected += 1
            return rand_move

        return False

    def played_game(self):
        self.num_games += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.reset_injected()

    def reset_injected(self):
        self.injected = 0
