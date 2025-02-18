"""
ReplayMemory.py

This file contains the ReplayMemory class.
"""
from collections import deque
import itertools
import random

class ReplayMemory():

  def __init__(self, ini, max_len=0, long_flag=False):
    random.seed(ini.get('random_seed'))
    self.ini = ini
    self.long_flag = long_flag
    if max_len != 0:
      # Memory is a simple deque where the elements are tuples:
      # (state, action, reward, next_state, done)
      self.memory = deque(maxlen=max_len)
    else:
      # Memory is a deque of lists of tuples. Each list represents a single game.
      if ini.get('model') == 'cnn' or ini.get('model') == 'rnn':
        self.memories = deque(maxlen=ini.get('max_memories'))
        self.cur_memory = []
      else: 
        self.memory = deque(maxlen=ini.get('max_memory'))

    if long_flag:
      self.memories = []
      self.cur_memory = []

  def __len__(self):
    return len(self.memory)

  def append(self, transition):
    # This is called from the AIAgent as:
    #
    #   self.memory.append((state, action, reward, next_state, done))
    #
    if self.ini.get('model') != 'cnn' and self.ini.get('model') != 'rnn':
      self.memory.append(transition)

    else:
      state, action, reward, next_state, done = transition
      if done:
        self.cur_memory.append(transition)
        self.memories.append(self.cur_memory)
        self.cur_memory = []
      else:
        self.cur_memory.append(transition)
        

      
  def get_memory(self):
    if self.ini.get('model') == 'cnn' or self.ini.get('model') == 'rnn':
      return random.sample(self.memories, 1)
    else:
      if len(self.memory) > self.batch_size:
        return random.sample(self.memory, self.batch_size)
      else:
        return self.memory
    
  def set_memory(self, memory):
    self.memory = memory
