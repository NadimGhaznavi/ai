"""
AIAgent.py

This class contains the AIAgent class. An instance of this class acts as
the player to the AISnakeGame.
"""
import os, sys
from collections import deque
from collections import namedtuple
import numpy as np
import random
import torch
from random import randint

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from QTrainer import QTrainer
from LinearQNet import LinearQNet
from SnakeGameElement import Direction
from SnakeGameElement import Point
from EpsilonAlgo import EpsilonAlgo
from NuAlgo import NuAlgo

class AIAgent:
  def __init__(self, ini, game):
    self.game = game
    self.ini = ini

    # Level 1 instances
    self.l1_epsilon_algo = EpsilonAlgo(ini, 1) # Epsilon Algorithm for exploration/exploitation
    self.l1_memory = deque(maxlen=ini.get('max_memory'))
    self.l1_model = LinearQNet(ini, 1)
    self.l1_trainer = QTrainer(ini, self.l1_model)

    self.l2_epsilon_algo = EpsilonAlgo(ini, 2)
    self.l2_memory = deque(maxlen=self.ini.get('max_memory'))
    self.l2_model = LinearQNet(ini, 2)
    self.l2_trainer = QTrainer(ini, self.l2_model)


    # Used in the state map, this initializes it to some random direction
    self.last_dirs = [ 0, 0, 1, 0 ]
    self.highscore = 0 # Current highscore
    self.batch_size = self.ini.get('batch_size')

    self.n_games = 0 # Number of games played
    self.n_games_buf = -1    
    
    self.load_checkpoint() # Load the simulation state from file if it exists
    self.save_highscore(0) # Initialize the highscore file

    # Create the simulation data directory if it does not exist
    os.makedirs(self.ini.get('sim_data_dir'), exist_ok=True)

  def get_action(self, state):

    l2_score = self.ini.get('l2_score')
    game_score = self.game.get_score()
    
    # Random epsilon based action (exploration)
    if game_score <= l2_score:
      # Use the Level 1 epsilon algorithm, with its own epsilon value
      random_move = self.l1_epsilon_algo.get_move()
    else:
      # Use the Level 2 epsilon algorithm, with its own epsilon value
      random_move = self.l2_epsilon_algo.get_move()
    
    if random_move:
      # Epsilon algorithm returned a move (not False)
      self.n_games_buf = self.n_games
      return random_move
    
    # AI agent based action
    final_move = [0, 0, 0]
    state0 = torch.tensor(state, dtype=torch.float)

    if game_score <= l2_score:
      # Use the Level 1 model
      prediction = self.l1_model(state0)
    else:
      # Use the Level 2 model
      prediction = self.l2_model(state0)

    move = torch.argmax(prediction).item()
    final_move[move] = 1 
      
    return final_move

  def get_checkpoint_filenames(self):
    # Get the checkpoint filename componenets
    ai_version = self.ini.get('ai_version')
    checkpoint_basename = self.ini.get('sim_checkpoint_basename')
    checkpoint_basename_l2 = self.ini.get('l2_sim_checkpoint_basename')
    sim_data_dir = self.ini.get('sim_data_dir')
    # Construct the file names
    checkpoint_file = os.path.join(sim_data_dir, str(ai_version) + checkpoint_basename)
    checkpoint_file_l2 = os.path.join(sim_data_dir, str(ai_version) + checkpoint_basename_l2)
    # Return the filenames
    return checkpoint_file, checkpoint_file_l2
  
  def get_model_filenames(self):
    # Get the model filename componenets
    ai_version = self.ini.get('ai_version')
    model_basename = self.ini.get('sim_model_basename')
    model_basename_l2 = self.ini.get('l2_sim_model_basename')
    sim_data_dir = self.ini.get('sim_data_dir')
    # Construct the file names
    model_file = os.path.join(sim_data_dir, str(ai_version) + model_basename)
    model_file_l2 = os.path.join(sim_data_dir, str(ai_version) + model_basename_l2)
    return model_file, model_file_l2

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
      ## Wall collision danger
      # Danger straight 
      (dir_r and game.is_wall_collision(point_r)),
      (dir_l and game.is_wall_collision(point_l)),
      (dir_u and game.is_wall_collision(point_u)),
      (dir_d and game.is_wall_collision(point_d)),

      # Danger right
      (dir_u and game.is_wall_collision(point_r)),
      (dir_d and game.is_wall_collision(point_l)),
      (dir_l and game.is_wall_collision(point_u)),
      (dir_r and game.is_wall_collision(point_d)),

      # Danger left
      (dir_d and game.is_wall_collision(point_r)),
      (dir_u and game.is_wall_collision(point_l)),
      (dir_r and game.is_wall_collision(point_u)),
      (dir_l and game.is_wall_collision(point_d)),

      ## Self collision danger
      # Danger straight
      (dir_r and game.is_snake_collision(point_r)),
      (dir_l and game.is_snake_collision(point_l)),
      (dir_u and game.is_snake_collision(point_u)),
      (dir_d and game.is_snake_collision(point_d)),

      # Danger right
      (dir_u and game.is_snake_collision(point_r)),
      (dir_d and game.is_snake_collision(point_l)),
      (dir_l and game.is_snake_collision(point_u)),
      (dir_r and game.is_snake_collision(point_d)),

      # Danger left
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

    # Include the previous direction of the snake in the state
    for aDir in self.last_dirs:
      state.append(aDir)
    self.last_dirs = [ dir_l, dir_r, dir_u, dir_d ]
    return np.array(state, dtype=int)

  def load_checkpoint(self):
    # Get the checkpoint filenames
    checkpoint_file, checkpoint_file_l2 = self.get_checkpoint_filenames()
    # Load the level 1 model checkpoint
    if os.path.isfile(checkpoint_file):
      self.l1_model.load_checkpoint(self.l1_trainer.optimizer, checkpoint_file)
      print(f"Loaded simulation checkpoint ({checkpoint_file})")
    # Repeat for the level 2 model checkpoint
    if os.path.isfile(checkpoint_file_l2):
      self.l2_model.load_checkpoint(self.l2_trainer.optimizer, checkpoint_file_l2)
      print(f"Loaded simulation checkpoint ({checkpoint_file_l2})")

  def load_model(self):
    model_file, model_file_l2 = self.get_model_filenames()
    # Load the level 1 model
    if os.path.isfile(model_file):
      self.l1_model.load_model(self.l1_trainer.optimizer, model_file)
      print(f"Loaded simulation model ({model_file})")
    # Repeat for the level 2 model
    if os.path.isfile(model_file_l2):
      self.l2_model.load_model(self.l2_trainer.optimizer, model_file_l2)
      print(f"Loaded simulation model ({model_file_l2})")

  def remember(self, state, action, reward, next_state, done):
    # Store the state, action, reward, next_state, and done in memory
    # Recall that memory is a deque, so it will automatically remove the oldest memory 
    # if the memory exceeds MAX_MEMORY
    if self.game.get_score() <= self.ini.get('l2_score'):
      # Use the level 1 memory
      self.l1_memory.append((state, action, reward, next_state, done))
    else:
      # User the level 2 memory
      self.l2_memory.append((state, action, reward, next_state, done))

  def save_checkpoint(self):
    # Get the checkpoint filename componenets
    checkpoint_file, checkpoint_file_l2 = self.get_checkpoint_filenames()
    # Save the simulation state
    if self.ini.get('sim_checkpoint_enable'):
      self.l1_model.save_checkpoint(self.l1_trainer.optimizer, checkpoint_file)
      self.l2_model.save_checkpoint(self.l2_trainer.optimizer, checkpoint_file_l2)
      if self.ini.get('sim_checkpoint_verbose'):
        print(f"Saved simulation checkpoint ({checkpoint_file})")
        print(f"Saved simulation checkpoint ({checkpoint_file_l2})")
  
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

  def save_model(self):
    # Get the model filenames
    model_file, model_file_l2 = self.get_model_filenames()
    # Save the simulation models
    self.l1_model.save_model(self.l1_trainer.optimizer, model_file)
    self.l2_model.save_model(self.l2_trainer.optimizer, model_file_l2)
    print(f"Saved simulation model ({model_file})")
    print(f"Saved simulation model ({model_file_l2})")

  def train_long_memory(self):
    l2_score = self.ini.get('l2_score')
    game_score = self.game.get_score()

    if game_score <= l2_score:
      # Use the level 1 memory
      if len(self.l1_memory) > self.batch_size:
        # Sample a random batch of memories
        mini_sample = random.sample(self.l1_memory, self.batch_size)
      else:
        mini_sample = self.l1_memory
      # Get the states, actions, rewards, next_states, and dones from the mini_sample
      states, actions, rewards, next_states, dones = zip(*mini_sample)
      self.l1_trainer.train_step(states, actions, rewards, next_states, dones)

    else:
      # Use the level 2 memory
      if len(self.l2_memory) > self.batch_size:
        # Sample a random batch of memories
        mini_sample = random.sample(self.l2_memory, self.batch_size)
      else:
        mini_sample = self.l2_memory
      # Get the states, actions, rewards, next_states, and dones from the mini_sample
      states, actions, rewards, next_states, dones = zip(*mini_sample)
      self.l2_trainer.train_step(states, actions, rewards, next_states, dones)

  def train_short_memory(self, state, action, reward, next_state, done):
    if self.game.score <= self.ini.get('l2_score'):
      self.l1_trainer.train_step(state, action, reward, next_state, done)
    else:
      self.l2_trainer.train_step(state, action, reward, next_state, done)
