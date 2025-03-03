"""
EpsilonAlgo.py

A class to encapsulate the functionality of the epsilon algorithm. The algorithm
injects random moves at the beginning of the simulation. The amount of moves
is controlled by the epsilon_value parameter which is in the AISnakeGame.ini and
can also be passed in when invoking the main asg.py front end.
"""
import random
from random import randint
import os, sys

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)

class EpsilonAlgo():
  def __init__(self, ini, log, stats):
    self.ini = ini
    self.log = log
    self.stats = stats
    # Set this random seed so things are repeatable
    random.seed(ini.get('random_seed')) 
    self.epsilon_value = ini.get('epsilon_value')
    self.epsilon_min = ini.get('epsilon_min')
    self.epsilon_decay = ini.get('epsilon_decay')
    if self.epsilon_value != 0:
      self.stats.set('epsilon', 'depleted', False)
    self.print_stats = ini.get('epsilon_print_stats')
    self.enabled = ini.get('epsilon_enabled')
    if not self.enabled:
      self.stats.set('epsilon', 'depleted', False)
    self.epsilon = self.epsilon_value
    self.num_games = 0
    self.injected = 0
    self.depleted = False
    
    if not self.enabled:
      self.ini.set('epsilon_print_stats', 'False')
    else:
      self.stats.set('epsilon', 'status', f'Epsilon greedy initialized with value of {self.epsilon_value}')

  def __str__(self):
    str_val = ''
    if self.epsilon > 0:
      str_val = 'injected# {:>3}, value {:>5}'.format(self.injected, self.epsilon)
    return str_val

  def get_move(self):
    if not self.enabled or self.epsilon == 0:
      return False
    
    if self.num_games % 2 == 0:
      return False
    
    if random.random() < self.epsilon:
      rand_move = [ 0, 0, 0 ]
      rand_idx = randint(0, 2)
      rand_move[rand_idx] = 1
      self.injected += 1
      return rand_move

    str_val = 'injected# {:>3}, value {:>5}'.format(self.injected, round(self.epsilon, 3))
    self.stats.set('epsilon', 'status', str_val)
    return False
  
  def played_game(self):
    self.num_games += 1
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) 
  
  def reset_injected(self):
    self.injected = 0

