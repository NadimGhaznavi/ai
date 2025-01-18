"""
NuAlgo.py

This class implements the nu algorithm. The algorithm injects random moves
whenever the AI reaches the current high score. This encourages the AI to
continue exploring as the snake grows in length. The number of random
moves is defined by the nu_value parameter. The current high score is tracked
by the nu_score variable. It has a default value of 1, but it can be increased
so that the nu algorithm is only triggered at higher scores.

The algorithm also includes some additional features.

If the AI plays nu_bad_games (e.g. 25) without getting a new high score then
the random pool is refilled and the pool_reset_count counter is incremented.

If the pool_reset_count is greater than 1 (e.g. if the AI has played 50, 
75, 100... games without getting a new high score) then the nu_score variable
is decreased by pool_reset_count. E.g. if nu_score is 25 and the AI has failed
to increase the highscore after 75, then the pool_reset count will be 3. The
nu_score will then decrease by 3. This causes the nu algorithm to inject random
moves at lower scores.

The values for nu_bad_games, nu_score and nu_value are in the AISnakeGame.ini
and can also be overriden with command line switches when you execute the main
asg.py front end script.
"""
import sys, os
from random import randint

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig

class NuAlgo():
  def __init__(self):
    # Constructor
    ini = AISnakeGameConfig()
    self.nu_bad_games = ini.get('nu_bad_games') # Number of games without a new high score
    self.nu_score = ini.get('nu_score') # Game score where algorithm is triggered
    self.nu_value = ini.get('nu_value') # Number of random moves in the nu pool
    self.cur_nu_value = self.nu_value # Current random pool size
    self.rand_moves_in_game = 0 # Number of random moves injected in last game
    self.pool_reset_count = 0  # How many times the pool was refilled without a highscore
    self.no_highscore = 0 # How many games were played with no highscore
    self.no_highscore_reset = 0 # How many times the "no_highscore so refill the pool" event happened
    print(f"NuAlgo: New instance created with threshold score of {self.nu_score}, a pool size of {self.nu_value} and a bad game threshold of {self.nu_bad_games}")

  def get_nu_score(self):
    return self.nu_score
  
  def get_nu_value(self):
    return self.cur_nu_value
  
  def get_pool_reset_count(self):
    return self.pool_reset_count

  def played_game(self):
    # Increment this counter
    self.no_highscore += 1
    # Check if the random move pool has been depleted
    if self.rand_moves_in_game:
      #pass
      print(f"NuAlgo: Injected {self.rand_moves_in_game} random moves into the last game, pool size is now {self.cur_nu_value}")
    self.rand_moves_in_game = 0
    if self.cur_nu_value <= 0:
      # Fill the random move pool back up
      print(f"NuAlgo: Random pool depleted, filling pool back up to {self.nu_value}")
      self.cur_nu_value = self.nu_value
      self.pool_reset_count += 1
      if self.pool_reset_count > 0:
        print(f"NuAlgo: Pool reset count is {self.pool_reset_count}")
      if self.pool_reset_count == 3:
        # Pool depleted, lower the threshold for injecting
        # random moves.
        self.nu_score -= 1
        print(f"NuAlog: Random pool depleted 3 times, decreasing threshold score to {self.nu_score}")
        self.pool_reset_count = 0

    if self.no_highscore == self.nu_bad_games:
      self.no_highscore_reset += 1
      bad_games = self.nu_bad_games * self.no_highscore_reset
      print(f"NuAlgo: AI played {bad_games} games without improvement, refilling the pool with {self.nu_value} random moves")
      self.cur_nu_value = self.nu_value
      if self.no_highscore_reset > 1:
        bad_games = self.no_highscore_reset * self.nu_bad_games
        print(f"NuAlgo: AI played {bad_games} games without improvement, decreasing the nu_score by {self.no_highscore_reset}")
        self.nu_score -= self.no_highscore_reset
        if self.nu_score < 1:
          # Make sure nu_score is *at least* 1. If this is triggered, you need to change other hyper parameters, because your
          # other settings are terrible. :)
          self.nu_score = 1

  def get_move(self, game_score):
    if game_score < self.nu_score:
      # NuAlgo is only triggered when the game score is
      # greater than or equal to the nu_score.
      return False
    
    # Random chance of making a new move
    rand_num = randint(0, self.nu_value)
    if rand_num >= self.cur_nu_value:
      # Nope, just return False and let the AI agent
      # make a move based on it's calculations
      return False

    # Track the number of random moves returned
    self.rand_moves_in_game += 1
    # Reduce the size of the nu random move pool since
    # we're about to return a random move
    self.cur_nu_value -= 1
    # Generate the random move
    rand_move = [ 0, 0, 0 ]
    rand_idx = randint(0, 2)
    rand_move[rand_idx] = 1
    # Return the random move
    return rand_move
  
  def new_highscore(self, high_score):
    # Set a new threshold 
    self.nu_score = high_score + 1
    # Fill up the nu random move pool
    self.cur_nu_value = self.nu_value
    self.pool_reset_count = 0
    # Reset these counters
    self.no_highscore = 0
    self.no_highscore_reset = 0
    print(f"NuAlgo: Threshold score increased to {self.nu_score}, random move pool reset to {self.nu_value}")

    