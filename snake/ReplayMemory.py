"""
ReplayMemory.py

This file contains the ReplayMemory class.
"""

from collections import deque
import random, sys


class MEM_TYPE:
    SHUFFLE = "shuffle"
    RANDOM_GAME = "random_game"


class ReplayMemory:

    def __init__(self):
        random.seed(1970)
        self.batch_size = 250
        # Valid options: shuffle, random_game, targeted_score, random_targeted_score
        self.mem_type = MEM_TYPE.SHUFFLE
        self.min_games = 1
        self.max_states = 15000
        self.max_shuffle_games = 40
        self.max_games = 500

        if self.mem_type == MEM_TYPE.SHUFFLE:
            # States are stored in a deque and a random sample will be returned
            self.memories = deque(maxlen=self.max_states)

        elif self.mem_type == MEM_TYPE.RANDOM_GAME:
            # All of the states for a game are stored, in order, in a deque.
            # A complete game will be returned
            self.memories = deque(maxlen=self.max_shuffle_games)
            self.cur_memory = []

        else:
            print(f"ERROR: Unrecognized replay memory type ({self.mem_type}), exiting")
            sys.exit(1)

    def append(self, transition):
        ## Add memories

        # States are stored in a deque and a random sample will be returned
        if self.mem_type == MEM_TYPE.SHUFFLE:
            self.memories.append(transition)

        # All of the states for a game are stored, in order, in a deque.
        # A set of ordered states representing a complete game will be returned
        elif self.mem_type == MEM_TYPE.RANDOM_GAME:
            self.cur_memory.append(transition)
            state, action, reward, next_state, done = transition
            if done:
                self.memories.append(self.cur_memory)
                self.cur_memory = []

    def get_random_game(self):
        if len(self.memories) >= self.min_games:
            rand_game = random.sample(self.memories, 1)
            return rand_game
        else:
            return False

    def get_random_states(self):
        mem_size = len(self.memories)
        if mem_size < self.batch_size:
            return self.memories
        return random.sample(self.memories, self.batch_size)

    def get_memory(self):

        if self.mem_type == MEM_TYPE.SHUFFLE:
            return self.get_random_states()

        elif self.mem_type == MEM_TYPE.RANDOM_GAME:
            return self.get_random_game()

    def set_memory(self, memory):
        self.memory = memory
