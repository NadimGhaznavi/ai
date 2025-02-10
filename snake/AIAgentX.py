"""
AIAgent.py

This class contains the AIAgent class. An instance of this class acts as
the player to the AISnakeGame.
"""
import os, sys
from collections import deque
from collections import namedtuple
import numpy as np
import torch

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from QTrainer import QTrainer
from QTrainerX import QTrainerX
from LinearQNet import LinearQNet
from ModelX import ModelX
from SnakeGameElement import Direction
from SnakeGameElement import Point
from EpsilonAlgo import EpsilonAlgo
from ReplayMemory import ReplayMemory
from NuAlgo import NuAlgo

class AIAgent:
  def __init__(self, ini, log, game):
    self.ini = ini
    self.log = log
    self.game = game
    self.min_score = ini.get('agent_x_min_score')

    # Level 0 initialization
    self.net = {}
    self.add_level()

    # Initialize Nu and Epsilon
    self.nu_algo = NuAlgo(ini, log)
    self.epsilon_algo = EpsilonAlgo(ini, log)

    # Used in the state map, this initializes it to some random direction
    self.last_dirs = [ 0, 0, 1, 0 ]
    self.highscore = 0 # Current highscore
    self.batch_size = self.ini.get('batch_size') # Size of the ReplayMemory's dequeu
    self.scores = {}

    self.n_games = 0 # Number of games played
    self.n_games_buf = -1    
    
    # Initialize the highscore
    self.highscore = 0
    self.save_highscore(self.highscore) 

    # Create the simulation data directory if it does not exist
    os.makedirs(self.ini.get('sim_data_dir'), exist_ok=True)

  def add_level(self):
    new_level_num = len(self.net)
    new_level = {
      'memory': ReplayMemory(self.ini),
      'model' : ModelX(self.ini, self.log, new_level_num),
      'num_games' : 0
    }
    new_level['trainer'] = QTrainerX(self.ini, new_level['model'], new_level_num)
    self.net[new_level_num] = new_level
    self.upgrade(new_level_num)
    
  def dream(self, level):
    # Get the last memory out
    lower_level = level - 1
    if lower_level <= self.min_score:
      lower_level = 0
    low_memory = self.net[lower_level]['memory']
    low_trainer = self.net[lower_level]['trainer']
    (state, action, reward, next_state, done) = low_memory.get_good_memory()

    if (state, action, reward, next_state, done):
      dream_reward = self.ini.get('agent_x_dream_reward')
      new_reward = dream_reward * reward
      low_memory.append((state, action, new_reward, next_state, done))
      # Execute a training step with the memory
      low_trainer.train_step(state, action, new_reward, next_state, done)
      self.log.log(f'L{level} Dreams: Sweet dream with {new_reward} ({level}/{lower_level})')
    

  def get_action(self, state):
    cur_score = self.game.get_score()
    # Use the epsilon algorithm; exploration
    random_move = self.epsilon_algo.get_move()
    # Use the nu algorithm; exploration
    random_move = self.nu_algo.get_move(cur_score)
    if random_move:
      # Random move was returned
      self.n_games_buf = self.n_games
      return random_move
    # AI agent based action
    final_move = [0, 0, 0]
    state0 = torch.tensor(state, dtype=torch.float)
    # Get the prediction
    if cur_score <= self.min_score:
      prediction = self.net[0]['model'](state0)
    else:
      prediction = self.net[cur_score]['model'](state0)
    move = torch.argmax(prediction).item()
    final_move[move] = 1 
    return final_move

  def get_epsilon(self):
    return self.epsilon_algo
  
  def get_epsilon_value(self):
    return self.epsilon_algo.get_epsilon_value()

  def get_highscore(self):
    return self.highscore

  def get_model_steps(self, score):
    return self.net[score]['model'].get_steps()

  def get_nu_algo(self):
    return self.nu_algo
  
  def get_num_games(self):
    return self.n_games
  
  def get_score(self):
    return self.game.get_score()
  
  def get_snake_length_in_binary(self):
    # Get the length of the snake in binary.
    # This is used in the state map, the get_state() function.
    bin_str = format(len(self.game.snake), 'b')
    out_list = []
    for bit in range(len(bin_str)):
      out_list.append(bin_str[bit])
    for zero in range(7 - len(out_list)):
      out_list.insert(0, '0')
    for x in range(7):
      out_list[x] = int(out_list[x])
    return out_list
  
  def get_state(self):
    # Returns the current state of the game.
    game = self.game
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    slb = self.get_snake_length_in_binary()

    state = [
      # Wall collision straight ahead
      (dir_r and game.is_wall_collision(point_r)),
      (dir_l and game.is_wall_collision(point_l)),
      (dir_u and game.is_wall_collision(point_u)),
      (dir_d and game.is_wall_collision(point_d)),

      # Wall collision to the right
      (dir_u and game.is_wall_collision(point_r)),
      (dir_d and game.is_wall_collision(point_l)),
      (dir_l and game.is_wall_collision(point_u)),
      (dir_r and game.is_wall_collision(point_d)),

      # Wall collision to the left
      (dir_d and game.is_wall_collision(point_r)),
      (dir_u and game.is_wall_collision(point_l)),
      (dir_r and game.is_wall_collision(point_u)),
      (dir_l and game.is_wall_collision(point_d)),

      # Snake collision straight ahead
      (dir_r and game.is_snake_collision(point_r)),
      (dir_l and game.is_snake_collision(point_l)),
      (dir_u and game.is_snake_collision(point_u)),
      (dir_d and game.is_snake_collision(point_d)),

      # Snake collision to the right
      (dir_u and game.is_snake_collision(point_r)),
      (dir_d and game.is_snake_collision(point_l)),
      (dir_l and game.is_snake_collision(point_u)),
      (dir_r and game.is_snake_collision(point_d)),

      # Snake collision to the left
      (dir_d and game.is_snake_collision(point_r)),
      (dir_u and game.is_snake_collision(point_l)),
      (dir_r and game.is_snake_collision(point_u)),
      (dir_l and game.is_snake_collision(point_d)),

      # Move direction
      dir_l, dir_r, dir_u, dir_d,

      # Food location
      game.food.x < game.head.x, # Food left
      game.food.x > game.head.x, # Food right
      game.food.y < game.head.y, # Food up
      game.food.y > game.head.y, # Food down
      game.food.x == game.head.x, # Food ahead or behind
      game.food.y == game.head.y, # Food above or below

      # Snake length in binary using 7 bits
      slb[0], slb[1], slb[2], slb[3], slb[4], slb[5], slb[6],
    ]

    # Previous direction of the snake
    for aDir in self.last_dirs:
      state.append(aDir)
    self.last_dirs = [ dir_l, dir_r, dir_u, dir_d ]
    return np.array(state, dtype=int)

  def get_trainer_steps(self, score):
    if score <= self.min_score:
      return self.net[0]['trainer'].get_steps()
    else:
      return self.net[score]['trainer'].get_steps()

  def increment_games(self):
    self.n_games += 1

  def log_loss(self):
    loss_str = 'AIAgentX Loss          '
    count = 0
    for level in self.net:
      loss = self.net[level]['trainer'].get_total_loss() / self.get_num_games()
      if count and count % 10 == 0:
        self.log.log(loss_str)
        loss_str = 'AIAgentX Loss + {:<2}     '.format(count)
      loss_str = loss_str + '{:>6.2f}  '.format(round(loss,2))
      count += 1
    self.log.log(loss_str)

  def log_scores(self):
    log_str = 'AIAgentX (v' + str(self.ini.get('ai_version')) + ')' + \
      ' Game: {:>4}'.format(self.get_num_games()) + \
      ', Score: {:>2}'.format(self.game.get_score()) + \
      ', Highscore: {:>2}'.format(self.get_highscore()) 
    self.log.log(log_str)
    self.log.log('AIAgentX Game #      {:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}'.format(0,1,2,3,4,5,6,7,8,9))
    self.log.log('AIAgentX' + (' ' * 17) + ('-' * 79))
    score_str = 'AIAgentX Count       '
    max_score = 0
    for score in self.scores.keys():
      if score > max_score:
        max_score = score
    while max_score > 0:
      if max_score not in self.scores:
        self.scores[max_score] = 0
      max_score -= 1
    # Sort the dictionary by score
    self.scores = dict(sorted(self.scores.items()))
    count = 0
    for score in self.scores.keys():
      if count and count % 10 == 0:
        self.log.log(score_str) # Chop off last ', ' 
        score_str = 'AIAgentX Count + {:<2}  '.format(count)
      score_str = score_str + '{:>8}'.format(self.scores[score])
      count += 1
    self.log.log(score_str) # Chop off last ', ' 
    self.log.log('AIAgentX' + (' ' * 17) + ('-' * 79))
  
  def log_upgrades(self, stats):
    #self.log.log('AIAgentX Game # {:<4} '.format(self.get_num_games()) + ('-' * 10) + \
    #             '| {:<12} |'.format('Upgrades') + ('-' * 57))
    upgrades_str = 'AIAgentX Upgrade     '
    count = 0
    self.upgrades = dict(sorted(self.scores.items()))
    count = 0
    # Fill in any blanks
    max_score = 0
    for score in stats:
      if count and count % 10 == 0:
        self.log.log(upgrades_str) # Chop off last ', ' 
        upgrades_str = 'AIAgentX Upgrade + {:<2}'.format(count)
      upgrades_str = upgrades_str + '{:>8}'.format(stats[score]['upgrades'])
      count += 1
    
    #if count % 10 != 0:
      # Make sure we don't log the same message twice when count == 10.
    self.log.log(upgrades_str) # Chop off last ', ' 
    self.log.log('AIAgentX' + (' ' * 17) + ('-' * 79))
 
  def played_game(self, score):
    self.epsilon_algo.played_game()
    self.nu_algo.played_game(score)
    if score in self.scores:
      self.scores[score] += 1
    else:
      self.scores[score] = 1

  def print_model(self):
    # All of the models share the same structure, return one of them
    print(self.net[0]['model'])

  def remember(self, state, action, reward, next_state, done):
    # Store the state, action, reward, next_state, and done in memory
    # Recall that memory is a deque, so it will automatically remove the oldest memory 
    # if the memory exceeds MAX_MEMORY
    score = self.get_score()
    # S
    if score <= self.min_score:
      self.net[0]['memory'].append((state, action, reward, next_state, done))
    else:
      self.net[score]['memory'].append((state, action, reward, next_state, done))

  def reset_epsilon_injected(self):
    self.epsilon_algo.reset_injected()
  def reset_nu_algo_injected(self):
    self.nu_algo.reset_injected()

  def reset_model_steps(self):
    self.net[0]['model'].reset_steps()

  def reset_trainer_steps(self):
    score = self.get_score()
    self.net[score]['trainer'].reset_steps()

  def save_checkpoint(self):
    # Save the models for each level
    print("NOT IMPLEMENTED")
  
  def save_highscore(self, highscore):
    # Get the highscore filename components
    sim_highscore_basename = self.ini.get('sim_highscore_basename')
    sim_data_dir = self.ini.get('sim_data_dir')
    # Construct the file name
    highscore_file = os.path.join(
      sim_data_dir, str(self.ini.get('ai_version')) + sim_highscore_basename)
    if not os.path.exists(highscore_file):
      # Create a new highscore file
      with open(highscore_file, 'w') as file_handle:
        file_handle.write("Game Number,High Score\n")
        file_handle.write("0,0\n")
    else:
      # Append the current game number and score to the highscore file
      with open(highscore_file, 'a') as file_handle:
        file_handle.write(str(self.n_games) + ',' + str(highscore) + "\n")

  def set_highscore(self, score):
    self.highscore = score
    self.save_highscore(score)

  def set_stats(self, stats):
    self.stats = stats

  def set_nu_algo_highscore(self, score):
    # We need to let *all* of the NuAlgo instances know about the new high score
    self.nu_algo.new_highscore(score)
    
  def share_dream(self, level):
    self.log.log(f"L{level} Dreams: Share dream ...")
    while level != 0:
      self.dream(level)
      level -= 1

  def train_long_memory(self):
    # Get the states, actions, rewards, next_states, and dones from the mini_sample
    score = self.get_score()

    if score <= self.min_score:
      # Route all scores below MIN_SCORE to the bottom layer
      mini_sample = self.net[0]['memory'].get_memory()
      states, actions, rewards, next_states, dones = zip(*mini_sample)
      self.net[0]['trainer'].train_step(states, actions, rewards, next_states, dones)
      
    else:
      while score > -1:
        # Train all networks below this network at the same time.
        mini_sample = self.net[score]['memory'].get_memory()
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.net[score]['trainer'].train_step(states, actions, rewards, next_states, dones)
        score -= 1

  def train_short_memory(self, state, action, reward, next_state, done):
    score = self.get_score()
    if score not in self.net:
      # Create a level for this new score
      self.add_level()
      # Make sure we get the training step in on the original level
      self.net[score - 1]['trainer'].train_step(state, action, reward, next_state, done)  

    # Route all training for the first 5 points to one bottom layer
    if score <= self.min_score:
      self.net[0]['trainer'].train_step(state, action, reward, next_state, done)
    else:
      self.net[score]['trainer'].train_step(state, action, reward, next_state, done)

  def upgrade(self, level, num_times=0):
    if level > self.min_score:
      tmp_file = str(self.ini.get('ai_version')) + '.tmp'
      old_level = level - 1
      if old_level == 3 or old_level == 2 or old_level == 1: 
        old_level = 0
      torch.save({
        'model_state_dict': self.net[old_level]['model'].state_dict(),
        'optimizer_state_dict': self.net[old_level]['trainer'].optimizer.state_dict(),
        'weights_only': False
      }, tmp_file)
      checkpoint = torch.load(tmp_file, weights_only=False)
      state_dict = checkpoint['model_state_dict']
      self.net[level]['model'].load_state_dict(state_dict)
      self.net[level]['trainer'].optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      os.remove(tmp_file)
      # Upgrade any level below us as well
      while level != 0:
        level -= 1
        self.upgrade(level, num_times)