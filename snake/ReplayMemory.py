"""
ReplayMemory.py

This file contains the ReplayMemory class.
"""
from collections import deque
import itertools
import random

class ReplayMemory():

  def __init__(self, ini, max_len=0):
    random.seed(ini.get('random_seed'))
    self.ini = ini
    if max_len != 0:
      self.memory = deque(maxlen=max_len)
    else:
      if ini.get('model') == 'cnn':
        self.memory = deque(maxlen=ini.get('batch_size'))
      else:
        self.memory = deque(maxlen=ini.get('max_memory'))
    self.batch_size = ini.get('batch_size')

  def __len__(self):
    return len(self.memory)

  def append(self, transition):
    self.memory.append(transition)

  def get_memory(self):
    if self.ini.get('model') == 'cnn':
      return self.memory
    else:
      if len(self.memory) > self.batch_size:
        return random.sample(self.memory, self.batch_size)
      else:
        return self.memory
    
  def set_memory(self, memory):
    self.memory = memory
