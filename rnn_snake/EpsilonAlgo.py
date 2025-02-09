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
    self.stats.set('epsilon', 'status', self)

    if not self.enabled:
      self.ini.set_value('epsilon_print_stats', 'False')
    else:
      self.log.log(f"EpsilonAlgo: New instance with epsilon value of {self.epsilon_value}")

  def __str__(self):
    str_val = ''
    if self.epsilon > 0:
      str_val = 'injected# {:>3}, units# {:>3}'.format(self.injected, self.epsilon)
    return str_val

  def get_move(self):
    if self.enabled and self.depleted == False and self.epsilon < 0:
      self.log.log(f"EpilsonAlgo: Pool has been depleted")
      self.depleted = True
    rand_num = randint(0, self.epsilon_value)
    if self.epsilon <= 0:
      self.stats.set('epsilon', 'depleted', True)
    if rand_num < self.epsilon:
      rand_move = [ 0, 0, 0 ]
      rand_idx = randint(0, 2)
      rand_move[rand_idx] = 1
      self.injected += 1
      return rand_move
    self.stats.set('epsilon', 'status', self)
    return False
  
  def played_game(self):
    self.num_games += 1
    self.epsilon = self.epsilon_value - self.num_games
    if self.epsilon < 0:
      self.epsilon = 0
  
  def reset_injected(self):
    self.injected = 0

