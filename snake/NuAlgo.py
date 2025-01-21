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
import random
from random import randint

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig
from collections import deque

DEBUG = True

class NuAlgo():
  def __init__(self):
    # Constructor
    ini = AISnakeGameConfig()
    # Set this random seed so things are repeatable
    random.seed(ini.get('random_seed')) 
    self.nu_value = ini.get('nu_value') # 60
    self.nu_score = ini.get('nu_score') # 1
    self.nu_bad_games = ini.get('nu_bad_games') # 25
    # Number of games in a row with a score of zero
    self.max_zero_scores = ini.get('nu_max_zero_scores') 
    self.bad_game_count = 0 # counter for nu_bad_games
    self.nu = self.nu_value # Size of the random move pool
    # Maximum number of random moves injected into a game
    self.max_moves = ini.get('nu_max_moves') 
    self.max_moves_count = 0 # Counter for max_moves
    # How many times the nu pool has been refilled without finding a high score
    self.nu_refill_count = 0
    self.enabled = ini.get('nu_enable') # Whether this algorithm is enabled
    self.print_stats = ini.get('nu_print_stats')
    self.injected = 0 # Number of random moves injected in a game
    self.game_scores = deque(maxlen=self.max_zero_scores)
    self.verbose = ini.get('nu_verbose')
    self.highscore_countdown = 0
    self.highscore_countdown_enable = False
    self.highscore_no_rand = 0

    if self.enabled == False:
      print("NuAlgo: NuAlgo is disabled")
      self.print_stats = False
      self.verbose = False
    else:
      if self.print_stats:
        print(f"NuAlgo: New instance with pool size ({self.nu_value}), score ({self.nu_score}) and bad games ({self.nu_bad_games})")

  def get_move(self, cur_score):
    if self.enabled == False:
      # NuAlgo is disabled
      return False
    
    if cur_score < self.nu_score:
      # Current game score too low to inject random moves
      return False
    
    rand_num = randint(0, self.nu_value)

    if rand_num >= self.nu:
      # No random move generated
      return False 
    
    # Give the AI 3 games to work with the new highscore before we
    # start injecting random moves with the new highscore.
    if self.highscore_countdown_enable == True and \
      self.highscore_countdown < self.highscore_no_rand:
      return False
    
    if self.highscore_countdown == self.highscore_no_rand:
      self.highscore_countdown = 0
      self.highscore_countdown_enable = False
    
    if self.max_moves_count < self.max_moves:
      self.max_moves_count += 1
      self.injected += 1
      rand_move = [ 0, 0, 0 ]
      rand_idx = randint(0, 2)
      rand_move[rand_idx] = 1
      # Reduce the size of the nu pool
      self.nu -= 1
      return rand_move

  def new_highscore(self, score):
    # There is a new high score
    self.highscore_countdown_enable = True
    self.highscore_countdown = 0
    self.nu_score = score
    self.bad_game_count = 0
    self.nu_refill_count = 0
    self.nu = self.nu_value
    if self.print_stats and self.verbose:
      print(f"NuAlgo: New high score, increasing score to ({score}) and refilling pool to ({self.nu})")

  def played_game(self, cur_score):
    # Increment every game (resets with a new highscore)
    if self.highscore_countdown_enable:
      self.highscore_countdown += 1

    self.bad_game_count += 1
    self.max_moves_count = 0
    self.cur_score = cur_score
    self.game_scores.append(cur_score)

    if cur_score > self.nu_score:
      # The current score is higher than the nu_score, use the current score
      self.nu_score = cur_score
      self.nu = self.nu_value
      self.nu_refill_count = 0
      self.bad_game_count = 0
      if self.print_stats and self.verbose:
        print(f"NuAlgo: Setting score to current score ({self.nu_score}), refilling pool to ({self.nu})")

    if self.nu_score == cur_score:
      # The current score has reached the nu_score, make sure we don't lower
      # the nu_score by a large number, so reset the nu_refill_count.
      if self.nu_refill_count > 1:
        self.nu_refill_count = 1
      
      if self.cur_score != 0:
        # Let's also give the AI a chance to get a new high score before
        # throwing more random moves into the mix, but not if the score
        # is zero, or NuAlgo will never be enabled.
        self.highscore_countdown_enable = True
        self.highscore_countdown = 0

    if self.bad_game_count == self.nu_bad_games and self.nu_refill_count == 0:
      # nu_bad_games without reaching nu_score
      # Increment this counter
      self.nu_refill_count += 1
      if self.print_stats and self.verbose:
        print(f"NuAlgo: Played ({self.bad_game_count}) games without reaching score ({self.nu_score}), incrementing reset count to ({self.nu_refill_count})")
      self.bad_game_count = 0

    elif self.bad_game_count == self.nu_bad_games and self.nu_refill_count == 1:
      # 2x nu_bad_games without reaching nu_score
      total_bad_games = self.nu_bad_games * 2
      if self.nu_score > 0:
        # Make sure we don't set nu_score below 0
        self.nu_score -= 1
      self.nu = self.nu_value
      if self.print_stats and self.verbose:
        print(f"NuAlgo: Played ({total_bad_games}) games without reaching score ({self.nu_score+1}), refilling pool to ({self.nu_value}), setting score to {self.nu_score}, incrementing reset count to ({self.nu_refill_count})")
      self.nu_refill_count += 1
      self.bad_game_count = 0

    elif self.bad_game_count == self.nu_bad_games and self.nu_refill_count > 1:
      # More than 2x (3, 4, 5...) nu_bad_games without reaching nu_score
      total_bad_games = self.bad_game_count * 3
      # Start dropping the nu_score by a faster factor: by self.nu_refill_count
      self.nu_score -= self.nu_refill_count
      if self.nu_score < 0:
        self.nu_score = 0
      self.nu = self.nu_value
      self.nu_refill_count += 1
      self.bad_game_count = 0
      if self.print_stats and self.verbose:
        print(f"NuAlgo: Played ({total_bad_games}) games without reaching score ({self.nu_score}), refilling pool to ({self.nu_value}), setting score to ({self.nu_score}), incrementing reset count to ({self.nu_refill_count})")

    elif len(self.game_scores) == self.max_zero_scores:
      # max_zero_scores is the number of games in a row with score zero, so instead
      # of slowly reducing nu_score, just set it to zero.
      all_zeros = True
      for score in self.game_scores:
        if score != 0:
          all_zeros = False
      if all_zeros:
        self.nu = self.nu_value
        self.nu_score = 0
        self.nu_refill_count = 0
        self.bad_game_count = 0
        self.game_scores = deque(maxlen=self.max_zero_scores)
        if self.print_stats and self.verbose:
          print(f"NuAlgo: Played ({self.max_zero_scores}) games with a score of zero, refilling pool to ({self.nu_value}) and setting score to ({self.nu_score})")
    
  def get_bad_game_count(self):
    return self.bad_game_count

  def get_injected(self):
    injected = self.injected
    self.injected = 0
    return injected

  def get_nu_bad_games(self):
    return self.nu_bad_games

  def get_nu_refill_count(self):
    return self.nu_refill_count

  def get_nu_score(self):
    return self.nu_score

  def get_nu_value(self):
    return self.nu


