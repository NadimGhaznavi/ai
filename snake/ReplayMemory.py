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
    
    self.enable_shuffle        = ini.get('replay_mem_enable_shuffle')
    self.enable_game_shuffle   = ini.get('replay_mem_enable_game_shuffle')
    self.enable_targeted_games = ini.get('replay_mem_enable_targeted_games')
    self.min_games             = ini.get('replay_mem_min_games')
    self.max_games             = ini.get('replay_mem_max_games')
    self.max_memories          = ini.get('replay_mem_max_games')
    self.min_memories          = ini.get('replay_mem_min_games')
    
    if self.enable_shuffle:
      # States are stored in a deque and a random sample will be returned
      self.batch_size = ini.get('replay_mem_batch_size')
      self.memories = deque(maxlen=ini.get('replay_mem_size'))
      
    elif self.enable_game_shuffle:
      # All of the states for a game are stored, in order, in a deque.
      # A complete game will be returned
      self.memories = deque(maxlen=self.max_games)
      self.cur_memory = []
      
    else:
      # All of the states for a games are stored in a dictionary, with the game score being the key
      # and the value being a deque of complete games for that score.
      # For enable_game shuffle, a random score is chosen and a random game for that score is returned.
      # For enable targetd_games, a game with the same score as the current score is returned.
      self.memories = {}
      self.extra_memories = {}
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
        # We have a deque for this score in self.memories
        self.memories[score].append(self.cur_memory)
        self.cur_memory = []
      
  def get_game_shuffle_memory(self):
    if len(self.memories) >= self.min_games:
      rand_game = random.sample(self.memories, 1)
      self.stats.set('replay', 'mem_size', len(rand_game[0]))
      return rand_game
    else:
      return False

  def get_shuffle_memory(self):
    mem_size = len(self.memories)
    if mem_size < self.batch_size:
      self.stats.set('replay', 'mem_size', mem_size)
      return self.memories
    else:
      self.stats.set('replay', 'mem_size', self.batch_size)
    return random.sample(self.memories, self.batch_size) 

  def get_targeted_game(self):
    score = self.stats.get('game', 'score') + 1
    if score in self.memories:
      rand_game = random.sample(self.memories[score], 1)
      msg = '{:>2} /{:>5}'.format(score, len(rand_game[0]))
      self.stats.set('replay', 'mem_size', msg)
      return rand_game
    self.stats.set('replay', 'mem_size', 'N/A')
    return False

  def get_memory(self):

    if self.enable_game_shuffle:
      return self.get_game_shuffle_memory()

    elif self.enable_shuffle:
      return self.get_shuffle_memory()

    elif self.enable_targeted_games:
      return self.get_targeted_game()

    valid_idxs = []
    for idx in self.memories.keys():
      if len(self.memories[idx]) >= self.min_memories:
        valid_idxs.append(idx)

    if len(valid_idxs) == 0:
      # No available games
      self.stats.set('replay', 'mem_size', 'N/A')
      return False

    rand_idx = random.choice(valid_idxs)
    rand_game = random.sample(self.memories[rand_idx], 1)
    msg = '{:>2} /{:>5}'.format(rand_idx, len(rand_game[0]))
    self.stats.set('replay', 'mem_size', msg)
    return rand_game

  def log_stats(self):
    if self.enable_game_shuffle:
      msg = f'Replay Memory (Game Shuffle) storing {len(self.memories)} games'

    elif self.enable_shuffle:
      msg = f'Replay Memory (Shuffle) storing {len(self.memories)} states'

    else:
      msg = f'Replay Memory (Ordered Shuffle)\n'
      scores = []
      for score in self.memories.keys():
        scores.append(score)
      scores.sort()
      msg += 'Score   Games\n'
      for score in scores:
        msg += '{:>5}   {}\n'.format(score, len(self.memories[score]))

      msg = msg[0:-1] # Chop off the last newline
    self.log.log(msg)
    
  def set_memory(self, memory):
    self.memory = memory

    
    

    