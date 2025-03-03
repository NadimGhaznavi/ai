"""
ReplayMemory.py

This file contains the ReplayMemory class.
"""
from collections import deque
import torch
import numpy as np
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
      model_type = ini.get('model')
      if model_type == 'cnn' or model_type == 'cnnr' or model_type == 't':
        self.memories = deque(maxlen=ini.get('max_memories'))
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
    if model_type != 'cnn' and \
      model_type != 'cnnr' and model_type != 't':
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
    model_type = self.ini.get('model')
    if model_type == 't':
      return random.sample(self.memories, 1)
    
    elif model_type == 'cnn' or model_type == 'cnnr':
      ran_game = random.sample(self.memories, 1)
      return ran_game
    
    elif model_type == 't':
      print("DEBUG len(self.memory): ", len(self.memory))
      if len(self.memory) > self.batch_size:
        memories = random.sample(self.memory, self.batch_size)
      else:
        memories = self.memory
      states, actions, rewards, next_states, dones = zip(*memories)
      states = torch.tensor(np.array(states), dtype=torch.float32).detach()
      next_states = torch.tensor(np.array(next_states), dtype=torch.float32).detach()
      #actions = torch.tensor(actions, dtype=torch.int64).detach()
      #if action.dim() > 0: # If action is a vector, get the index
      #    action = torch.argmax(action).long()
      #rewards = torch.tensor(rewards, dtype=torch.float32).detach()
      dones = torch.tensor(dones, dtype=torch.bool).detach()

      return states, actions, rewards, next_states, dones


    else:
      if len(self.memory) > self.batch_size:
        return random.sample(self.memory, self.batch_size)
      else:
        return self.memory
    
  def set_memory(self, memory):
    self.memory = memory
