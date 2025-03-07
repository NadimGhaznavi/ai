"""
ReplayMemory.py

This file contains the ReplayMemory class.
"""
from collections import deque
import random

class ReplayMemory():

  def __init__(self, ini, log, stats, max_len=0):
    random.seed(ini.get('random_seed'))
    self.ini = ini
    self.stats = stats
    self.log = log
    self.memories = {}
    self.cur_memory = []

  def append(self, transition):
    # This is called from the AIAgent as:
    #
    #   self.memory.append((state, action, reward, next_state, done))
    #
    state, action, reward, next_state, done = transition
    if done:
      score = self.stats.get('game', 'score')
      if score > 0:
        if score not in self.memories:
          self.memories[score] = deque(maxlen=self.ini.get('max_memories'))
        self.cur_memory.append(transition)
        self.memories[score].append(self.cur_memory)
      self.cur_memory = []
    else:
      self.cur_memory.append(transition)
      
  def get_memory(self):
    if len(self.memories.keys()) > 0:
      valid_idxs = []
      for idx in self.memories.keys():
        if len(self.memories[idx]) > 9:
          valid_idxs.append(idx)
      if len(valid_idxs) == 0:
        return False
      idx = random.choice(valid_idxs)
      ran_game = random.sample(self.memories[idx], 1)
      moves = len(ran_game[0])
      msg = f"Training on game with score {idx} and {moves} moves"
      self.log.log(msg)
      return ran_game
    else:
      return False
    
  def set_memory(self, memory):
    self.memory = memory
