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
    self.print_stats = ini.get('epsilon_print_stats')
    self.enabled = ini.get('epsilon_enabled')
    if not self.enabled:
      self.stats.set('epsilon', 'depleted', False)
    if self.epsilon_value != 0:
      self.stats.set('epsilon', 'depleted', False)
    self.epsilon = self.epsilon_value
    self.num_games = 0
    self.injected = 0
    self.depleted = False
    
    if not self.enabled:
      self.ini.set('epsilon_print_stats', 'False')

  def get_move(self):
    if not self.enabled or self.epsilon == 0:
      return False
    
    if random.random() < self.epsilon:
      rand_move = [ 0, 0, 0 ]
      rand_idx = randint(0, 2)
      rand_move[rand_idx] = 1
      self.injected += 1
      self.stats.set('epsilon', 'injected', self.injected)
      return rand_move

    return False
  
  def played_game(self):
    self.num_games += 1
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) 
    self.stats.set('epsilon', 'value', self.epsilon)
    self.reset_injected()
  
  def reset_injected(self):
    self.injected = 0

