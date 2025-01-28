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
from LinearQNet import LinearQNet
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
    
    level_score = self.ini.get('level_score')

    # Level 1 initialization
    self.level = { 
      level_score: {
        'memory': ReplayMemory(ini),
        'model': LinearQNet(ini, log, level_score),
      }
    }
    model = self.level[level_score]['model']
    self.level[level_score]['trainer'] = QTrainer(ini, model, level_score)

    self.nu_algo = NuAlgo(ini, log)
    self.epsilon_algo = EpsilonAlgo(ini, log)

    # Used in the state map, this initializes it to some random direction
    self.last_dirs = [ 0, 0, 1, 0 ]
    self.highscore = 0 # Current highscore
    self.batch_size = self.ini.get('batch_size')

    self.n_games = 0 # Number of games played
    self.n_games_buf = -1    
    
    self.save_highscore(0) # Initialize the highscore file

    # Create the simulation data directory if it does not exist
    os.makedirs(self.ini.get('sim_data_dir'), exist_ok=True)

  def add_level(self, score):
    # Create a new level, complete with model, trainer, and epsilon/nu algorithms
    model_level = score // 10 * 10 + 10
    self.level[model_level] = {
      'memory': ReplayMemory(self.ini),
      'model': LinearQNet(self.ini, self.log, model_level),
    }
    model = self.level[model_level]['model']
    self.level[model_level]['trainer'] = QTrainer(self.ini, model, model_level)
    # Copy the previous model weights to the new model
    tmp_file = str(self.ini.get('ai_version')) + '_tmp.pt'
    optimizer = self.level[model_level - 10]['trainer'].optimizer
    self.level[model_level - 10]['model'].save_checkpoint(optimizer, tmp_file)
    self.level[model_level]['model'].restore_model(optimizer, tmp_file)
    os.remove(tmp_file)

  def get_action(self, state):

    # Use the epsilon algorithm; exploration
    random_move = self.epsilon_algo.get_move()
    
    # Use the nu algorithm; exploration
    random_move = self.nu_algo.get_move(self.game.get_score())

    if random_move:
      # Random move was returned
      self.n_games_buf = self.n_games
      return random_move
    
    # AI agent based action
    final_move = [0, 0, 0]
    state0 = torch.tensor(state, dtype=torch.float)

    # Get the prediction
    model_num = self.game.get_score() // 10 * 10 + 10
    prediction = self.level[model_num]['model'](state0)
    move = torch.argmax(prediction).item()
    final_move[move] = 1 
    return final_move

  def get_epsilon(self):
    return self.epsilon_algo
  
  def get_epsilon_value(self):
    return self.epsilon_algo.get_epsilon_value()

  def get_all_steps(self):
    all_steps = ''
    for model_num in self.level:
      all_steps = all_steps + 'Game# {:>5}, '.format(self.n_games)
      all_steps = all_steps + self.level[model_num]['model'].get_steps() + ', '
      all_steps = all_steps + self.level[model_num]['trainer'].get_steps() + '\n'
    return all_steps[:-1] # Chop off the trailing newline
  def get_model_steps(self, score):
    model_num = score // 10 * 10 + 10
    return self.level[model_num]['model'].get_steps()

  def get_nu_algo(self):
    return self.nu_algo
  
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

      # Snake length in binary using 7 bits
      slb[0], slb[1], slb[2], slb[3], slb[4], slb[5], slb[6],
    ]

    # Previous direction of the snake
    for aDir in self.last_dirs:
      state.append(aDir)
    self.last_dirs = [ dir_l, dir_r, dir_u, dir_d ]
    return np.array(state, dtype=int)

  def get_trainer_steps(self, score):
    model = score // 10 * 10 + 10
    return self.level[model]['trainer'].get_steps()

  def increment_games(self):
    self.n_games += 1

  def played_game(self, score):
    self.epsilon_algo.played_game()
    self.nu_algo.played_game(score)

  def print_model(self):
    # All of the models share the same structure, return one of them
    print(self.level[10]['model'])

  def remember(self, state, action, reward, next_state, done):
    # Store the state, action, reward, next_state, and done in memory
    # Recall that memory is a deque, so it will automatically remove the oldest memory 
    # if the memory exceeds MAX_MEMORY
    score = self.game.get_score()
    model_num = score // 10 * 10 + 10
    self.level[model_num]['memory'].append((state, action, reward, next_state, done))

  def reset_epsilon_injected(self):
    self.epsilon_algo.reset_injected()
  def reset_nu_algo_injected(self):
    self.nu_algo.reset_injected()

  def reset_model_steps(self, score):
    model = score // 10 * 10 + 10
    self.level[model]['model'].reset_steps()

  def reset_trainer_steps(self, score):
    model = score // 10 * 10 + 10
    self.level[model]['trainer'].reset_steps()

  def save_checkpoint(self):
    # Save the models for each level
    if self.ini.get('sim_checkpoint_enable'):
      data_dir = self.ini.get('sim_data_dir')
      checkpoint_basename = self.ini.get('sim_checkpoint_basename')
      for level in self.level:
        checkpoint_file = os.path.join(data_dir, str(self.ini.get('ai_version')) + '_L' + str(level) + checkpoint_basename)
        optimizer = self.level[level]['trainer'].optimizer
        self.level[level]['model'].save_checkpoint(optimizer, checkpoint_file)
        if self.ini.get('sim_checkpoint_verbose'):
          self.log.log(f"Saved simulation checkpoint ({checkpoint_file})")
  
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

  def set_nu_algo_highscore(self, score):
    # We need to let *all* of the NuAlgo instances know about the new high score
    self.nu_algo.new_highscore(score)
    
  def train_long_memory(self):
    game_score = self.game.get_score()
    model_num = game_score // 10 * 10 + 10
    count = 0

    while count != model_num:
      count += 10
      mini_sample = self.level[count]['memory'].get_memory()
      # Get the states, actions, rewards, next_states, and dones from the mini_sample
      states, actions, rewards, next_states, dones = zip(*mini_sample)
      self.level[count]['trainer'].train_step(states, actions, rewards, next_states, dones)

  def train_short_memory(self, state, action, reward, next_state, done):
    game_score = self.game.get_score()
    model_num = game_score // 10 * 10 + 10
    if model_num not in self.level:
      self.add_level(game_score)
    count = 0
    while count != model_num:
      count += 10
      self.level[count]['trainer'].train_step(state, action, reward, next_state, done)
