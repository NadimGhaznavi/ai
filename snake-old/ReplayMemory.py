"""
ReplayMemory.py

This file contains the ReplayMemory class.
"""
from collections import deque
import random, sys

class ReplayMemory():

  def __init__(self, ini, log, stats, max_len=0):
    random.seed(ini.get('random_seed'))
    self.ini = ini
    self.stats = stats
    self.log = log
    
    self.batch_size        = ini.get('replay_mem_batch_size')
    self.mem_type          = ini.get('replay_mem_type')
    self.min_games         = ini.get('replay_mem_min_games')
    self.max_states        = ini.get('replay_mem_max_states')
    self.max_shuffle_games = ini.get('replay_mem_max_shuffle_games')
    self.max_games         = ini.get('replay_mem_max_games')
    
    if self.mem_type == 'shuffle':
      # States are stored in a deque and a random sample will be returned
      self.memories = deque(maxlen=self.max_states)
      
    elif self.mem_type == 'random_game':
      # All of the states for a game are stored, in order, in a deque.
      # A complete game will be returned
      self.memories = deque(maxlen=self.max_shuffle_games)
      self.cur_memory = []
      
    elif self.mem_type == 'targeted_score' or self.mem_type == 'random_targeted_score':
      # All of the states for a games are stored in a dictionary, with the game score being the key
      # and the value being a deque of complete games for that score.
      # For enable_game shuffle, a random score is chosen and a random game for that score is returned.
      # For enable targetd_games, a game with the same score as the current score is returned.
      self.memories = {}
      self.cur_memory = []
    
    else:
      print("ERROR: Unrecognized replay memory type (" + self.mem_type + "), exiting")
      sys.exit(1)

  def append(self, transition):
    ## Add memories
    
    # States are stored in a deque and a random sample will be returned
    if self.mem_type == 'shuffle':
      self.memories.append(transition)

    # All of the states for a game are stored, in order, in a deque.
    # A set of ordered states representing a complete game will be returned
    elif self.mem_type == 'random_game':
      self.cur_memory.append(transition)
      state, action, reward, next_state, done = transition
      if done:
        self.memories.append(self.cur_memory)
        self.cur_memory = []

    elif self.mem_type == 'targeted_score' or self.mem_type == 'random_targeted_score':
      # All of the states for a games are stored in a dictionary, with the game score being the key
      # and the value being a deque of complete games for that score.
      # A random score is chosen and a random game for that score is returned.
      self.cur_memory.append(transition)
      state, action, reward, next_state, done = transition
      if done:
        score = self.stats.get('game', 'score')
        if score not in self.memories:
          self.memories[score] = deque(maxlen=self.max_games)
        # We have a deque for this score in self.memories
        self.memories[score].append(self.cur_memory)
        self.cur_memory = []

    else:
      print("ERROR: Unrecognized replay memory type (" + self.mem_type + "), exiting")
      sys.exit(1)
      
      
  def get_random_game(self):
    if len(self.memories) >= self.min_games:
      rand_game = random.sample(self.memories, 1)
      self.stats.set('replay', 'mem_size', len(rand_game[0]))
      return rand_game
    else:
      return False

  def get_random_states(self):
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
  
    score -= 2
    if score >= 0 and score in self.memories:
      rand_game = random.sample(self.memories[score], 1)
      msg = '{:>2} /{:>5}'.format(score - 1, len(rand_game[0]))
      self.stats.set('replay', 'mem_size', msg)
      return rand_game
    self.stats.set('replay', 'mem_size', 'N/A')
    return False

  def get_random_targeted_game(self):
    valid_idxs = []
    for idx in self.memories.keys():
      if len(self.memories[idx]) >= self.min_games:
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

  def get_memory(self):

    if self.mem_type == 'shuffle':
      return self.get_random_states()

    elif self.mem_type == 'random_game':
      return self.get_random_game()

    elif self.mem_type == 'targeted_score':
      return self.get_targeted_game()

    elif self.mem_type == 'random_targeted_score':
      return self.get_random_targeted_game()

    else:
      print("ERROR: Unrecognized replay memory type (" + self.mem_type + "), exiting")
      sys.exit(1)

  def log_stats(self):
    if self.mem_type == 'shuffle':
      msg = f'Replay Memory (Shuffle) storing {len(self.memories)} states'

    elif self.mem_type == 'random_game':
      msg = f'Replay Memory (Game Shuffle) storing {len(self.memories)} games'

    elif self.mem_type == 'targeted_score' or self.mem_type == 'random_targeted_score':
      msg = f'Replay Memory (Targeted)\n'
      scores = []
      for score in self.memories.keys():
        scores.append(score)
      scores.sort()
      msg += 'Score   Games\n'
      for score in scores:
        msg += '{:>5}   {}\n'.format(score, len(self.memories[score]))
      msg = msg[0:-1] # Chop off the last newline
      
    else:
      print("ERROR: Unrecognized replay memory type (" + self.mem_type + "), exiting")
      sys.exit(1)
      
    self.log.log(msg)
    
  def set_memory(self, memory):
    self.memory = memory

    
    

    