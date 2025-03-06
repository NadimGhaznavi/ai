"""
ReplayMemory.py

This file contains the ReplayMemory class.
"""
from collections import deque
import random

class ReplayMemory():

  def __init__(self, ini, stats, max_len=0, long_flag=False):
    random.seed(ini.get('random_seed'))
    self.ini = ini
    self.stats = stats
    self.long_flag = long_flag
    if max_len != 0:
      # Memory is a simple deque where the elements are tuples:
      # (state, action, reward, next_state, done)
      self.memory = deque(maxlen=max_len)
    else:
      # Memory is a deque of lists of tuples. Each list represents a single game.
      model_type = ini.get('model')
      if model_type == 'cnn' or model_type == 'cnnr':
        #self.memories = deque(maxlen=ini.get('max_memories'))
        self.memories = {}
        self.cur_memory = []
      else: 
        self.memory = deque(maxlen=ini.get('max_memory'))
        self.batch_size = ini.get('batch_size')


  def __len__(self):
    return len(self.memory)

  def append(self, transition):
    # This is called from the AIAgent as:
    #
    #   self.memory.append((state, action, reward, next_state, done))
    #
    model_type = self.ini.get('model')
    if model_type != 'cnn' and model_type != 'cnnr':
      self.memory.append(transition)

    else:
      state, action, reward, next_state, done = transition
      if done:
        score = self.stats.get('game', 'score')
        if score > 0:
          if score not in self.memories:
            self.memories[score] = deque(maxlen=self.ini.get('max_memory'))
          self.cur_memory.append(transition)
          self.memories[score].append(self.cur_memory)
        self.cur_memory = []
      else:
        self.cur_memory.append(transition)
      
  def get_memory(self):
    model_type = self.ini.get('model')
    if model_type == 'cnn' or model_type == 'cnnr':
      if len(self.memories.keys()) > 0:
        idx = random.choice(list(self.memories.keys()))
        if len(self.memories[idx]) < 10:
          return False
        ran_game = random.sample(self.memories[idx], 1)
        return ran_game
      else:
        return False
    
    else:
      if len(self.memory) > self.batch_size:
        return random.sample(self.memory, self.batch_size)
      else:
        return self.memory
    
  def set_memory(self, memory):
    self.memory = memory
