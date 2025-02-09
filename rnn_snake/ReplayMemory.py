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

  def __len__(self):
    return len(self.memory)

  def append(self, transition):
    self.memory.append(transition)

  def copy_memory(self):
    return self.memory.copy()

  def get_good_memory(self):
    for (state, action, reward, next_state, done) in self.memory:
      if reward > 0:
        return (state, action, reward, next_state, done)
    return False

  def get_memory(self):
    if len(self.memory) > self.batch_size:
      short_memory = deque([], maxlen=self.batch_size)
      count = 0
      while count < self.batch_size:
        short_memory.append(self.memory[count])
        count += 1  
      return short_memory
    else:
      return self.memory
    
  def pop(self):
    return self.memory.pop()
  
  def set_memory(self, memory):
    self.memory = memory
