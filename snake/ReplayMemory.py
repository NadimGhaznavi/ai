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
    self.enable_shuffle = ini.get('replay_mem_enable_shuffle')
    self.enable_game_shuffle = ini.get('replay_mem_enable_game_shuffle')
    if self.enable_shuffle:
      self.batch_size = ini.get('replay_mem_batch_size')
      self.memories = deque(maxlen=ini.get('replay_mem_size'))
    elif self.enable_game_shuffle:
      self.max_games=ini.get('replay_mem_max_games')
      self.memories = deque(maxlen=self.max_games)
      self.cur_memory = []
      self.max_memories = ini.get('replay_mem_max_games')
    else:
      self.max_memories = ini.get('replay_mem_max_games')
      self.memories = {}
      self.cur_memory = []

  def append(self, transition):
    # This is called from the AIAgent as:
    #
    #   self.memory.append((state, action, reward, next_state, done))
    #
    if self.enable_shuffle:
      # States are stored in a deque and a random sample will be returned
      self.memories.append(transition)

    elif self.enable_game_shuffle:
      # All of the states for a game are stored, in order, in a deque.
      # A complete game will be returned
      self.cur_memory.append(transition)
      state, action, reward, next_state, done = transition
      if done:
        self.memories.append(self.cur_memory)
        self.cur_memory = []

    else:
      # All of the states for a games are stored in a dictionary, with the game score being the key
      # and the value being a deque of complete games for that score.
      # A random score is chosen and a random game for that score is returned.
      self.cur_memory.append(transition)
      state, action, reward, next_state, done = transition
      if done:
        score = self.stats.get('game', 'score')
        if score not in self.memories:
          self.memories[score] = deque(maxlen=self.max_memories)
        self.memories[score].append(self.cur_memory)
        self.cur_memory = []
      
  def get_memory(self):

    if self.enable_game_shuffle:
      if len(self.memories) == self.max_games:
        ran_game = random.sample(self.memories, 1)
        self.stats.set('replay', 'mem_size', len(ran_game[0]))
        return ran_game
      else:
        return False

    elif self.enable_shuffle:
      mem_size = len(self.memories)
      if mem_size < self.batch_size:
        self.stats.set('replay', 'mem_size', mem_size)
        return self.memories
      else:
        self.stats.set('replay', 'mem_size', self.batch_size)
      return random.sample(self.memories, self.batch_size) 
    
    else:
      if len(self.memories.keys()) > 0:
        valid_idxs = []
        for idx in self.memories.keys():
          if len(self.memories[idx]) == self.max_memories:
            valid_idxs.append(idx)
        if len(valid_idxs) == 0:
          return False
        idx = random.choice(valid_idxs)
        ran_game = random.sample(self.memories[idx], 1)
        moves = len(ran_game[0])
        msg = '{:>2},{:>4}'.format(idx, moves)
        self.stats.set('replay', 'mem_size', msg)
        return ran_game
      else:
        return False
    
  def set_memory(self, memory):
    self.memory = memory

    
    

    