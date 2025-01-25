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
from AISnakeGameConfig import AISnakeGameConfig

class EpsilonAlgo():
  def __init__(self, ini, level):
    self.ini = ini
    # Set this random seed so things are repeatable
    random.seed(ini.get('random_seed')) 

    if level == 1:
      # This instance is for the Level 1 neural network
      self.epsilon_value = ini.get('epsilon_value')
    else:
      # This instance is for the Level 2 neural network
      self.epsilon_value = ini.get('l2_epsilon_value')

    self.print_stats = ini.get('epsilon_print_stats')
    
    self.epsilon = self.epsilon_value
    
    self.num_games = 0
    self.injected = 0
    self.depleted = False
    self.level = level
    
    if self.epsilon_value == 0:
      print(f"EpsilonAlgo({level}): EpsilonAlgo is disabled")
      # If the epsilon value is set to zero, disable the algorithm and don't print epsilon stats
      self.ini.set_value('epsilon_print_stats', 'False')
    else:
      print(f"EpsilonAlgo({level}): New instance with epsilon value of {self.epsilon_value}")

  def get_epsilon(self):
    if self.epsilon < 0:
      return 0
    return self.epsilon

  def get_epsilon_value(self):
    return self.epsilon_value
  
  def get_injected(self):
    injected = self.injected
    self.injected = 0
    return injected
  
  def get_move(self):
    rand_num = randint(0, self.epsilon_value)
    if self.epsilon < 0 and self.depleted == False:
      print(f"EpilsonAlgo: Model ({self.level}): Epsilon pool has been depleted")
      self.depleted = True

    if rand_num < self.epsilon:
      rand_move = [ 0, 0, 0 ]
      rand_idx = randint(0, 2)
      rand_move[rand_idx] = 1
      self.injected += 1
      return rand_move
    return False
  
  def get_print_stats(self):
    return self.ini.get('epsilon_print_stats')

  def played_game(self):
    self.num_games += 1
    self.epsilon = self.epsilon_value - self.num_games


