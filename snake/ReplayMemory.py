"""
ReplayMemory.py

This file contains the ReplayMemory class.
"""
from collections import deque
import random

class ReplayMemory():

  def __init__(self, ini, max_len=0):
    random.seed(ini.get('random_seed'))
    if max_len != 0:
      self.memory = deque(maxlen=max_len)
    else:
      self.memory = deque(maxlen=ini.get('max_memory'))
    self.batch_size = ini.get('batch_size')

  def append(self, transition):
    self.memory.append(transition)

  def get_memory(self):
    if len(self.memory) > self.batch_size:
      return random.sample(self.memory, self.batch_size)
    else:
      return self.memory
    
  def pop(self):
    return self.memory.pop()
