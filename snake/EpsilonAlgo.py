"""
EpsilonAlgo.py

A class to encapsulate the functionality of the epsilon algorithm. The algorithm
injects random moves at the beginning of the simulation. The amount of moves
is controlled by the epsilon_value parameter which is in the AISnakeGame.ini and
can also be passed in when invoking the main asg.py front end.
"""
from random import randint
import os, sys

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig

class EpsilonAlgo():
  def __init__(self):
    ini = AISnakeGameConfig()
    self.epsilon_value = ini.get('epsilon_value')
    self.epsilon = self.epsilon_value
    self.num_games = 0
    
    self.rand_moves_in_game = 0
    print(f"EpsilonAlgo: New instance with epsilon value of {self.epsilon_value}")

  def get_epsilon(self):
    return self.epsilon

  def played_game(self):
    self.num_games += 1
    self.epsilon = self.epsilon_value - self.num_games
    self.rand_moves_in_game = 0

  def get_move(self):
    rand_num = randint(0, self.epsilon_value)
    if rand_num < self.epsilon:
      rand_move = [ 0, 0, 0 ]
      rand_idx = randint(0, 2)
      rand_move[rand_idx] = 1
      self.rand_moves_in_game += 1
      return rand_move
    return False
