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
import configparser

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig
from QTrainer import QTrainer
from SnakeGameElement import Direction
from SnakeGameElement import Point
from AISnakeGameUtils import get_sim_desc

class AIAgent:
  def __init__(self, game, model, config, ai_version):
    ini = AISnakeGameConfig()
    self.game = game
    self.model = model
    self.config = config
    self.ai_version = ai_version

    self.batch_size = ini.get('batch_size')
    self.epsilon_value = ini.get('epsilon_value') # Epsilon value, for exploration (i.e. vs exploitation)
    self.gamma = ini.get('discount') # Discount rate, for future rewards
    self.highscore = 0

    self.last_dirs = [ 0, 0, 1, 0 ]
    self.learning_rate = ini.get('learning_rate')
    self.max_games = ini.get('max_games')
    self.max_score = ini.get('max_score')
    self.max_score_num = ini.get('max_score_num')
    self.max_score_num_count = 0
    self.memory = deque(maxlen=ini.get('max_memory'))
    self.n_games = 0 # Number of games played
    self.random_move_count = 0
    self.sim_checkpoint_basename = ini.get('sim_checkpoint_basename')
    self.sim_data_dir = ini.get('sim_data_dir')
    self.sim_highscore_basename = ini.get('sim_highscore_basename')
    self.sim_model_basename = ini.get('sim_model_basename')
    self.sim_desc_basename = ini.get('sim_desc_basename')
    self.trainer = QTrainer(self.model)

    self.nu_value = ini.get('nu_value')
    self.initial_nu_value = self.nu_value
    self.nu_score = ini.get('nu_score')
    self.initial_nu_score = self.nu_score
    self.nu_count = 0
    self.nu_num_games_same_score = 1

    self.load_checkpoint() # Load the simulation state from file if it exists
    self.save_highscore(0) # Save the "game #, highscore" metrics
    
  def get_snake_length_in_binary(self):
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
      (dir_d and game.is_snake_collision(point_r)) or
      (dir_u and game.is_snake_collision(point_l)) or
      (dir_r and game.is_snake_collision(point_u)) or
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
    # Include the previous direction of the snake
    for aDir in self.last_dirs:
      state.append(aDir)
    self.last_dirs = [ dir_l, dir_r, dir_u, dir_d ]
    return np.array(state, dtype=int)

  def load_checkpoint(self):
    checkpoint_file = str(self.ai_version) + self.sim_checkpoint_basename
    checkpoint_file = os.path.join(self.sim_data_dir, checkpoint_file)
    if os.path.isfile(checkpoint_file):
      optimizer = self.trainer.optimizer
      self.model.load_checkpoint(optimizer, checkpoint_file)
      print(f"Loaded simulation checkpoint ({checkpoint_file})")

  def load_model(self):
    model_file = str(self.ai_version) + self.sim_model_basename
    model_file = os.path.join(self.sim_data_dir, model_file)
    if os.path.isfile(model_file):
      optimizer = self.trainer.optimizer
      self.model.load_model(optimizer, model_file)
      print(f"Loaded simulation model ({model_file})")

  def remember(self, state, action, reward, next_state, done):
    # Store the state, action, reward, next_state, and done in memory
    # Recall that memory is a deque, so it will automatically remove the oldest memory 
    # if the memory exceeds MAX_MEMORY
    self.memory.append((state, action, reward, next_state, done))

  def save_checkpoint(self):
    # Save the simulation state
    checkpoint_file = str(self.ai_version) + self.sim_checkpoint_basename
    checkpoint_file = os.path.join(self.sim_data_dir, checkpoint_file)
    if not os.path.exists(self.sim_data_dir):
      os.makedirs(self.sim_data_dir)
    self.model.save_checkpoint(self.trainer.optimizer, checkpoint_file, self.game.num_games)
    print(f"Saved simulation checkpoint ({checkpoint_file})")
    self.save_sim_desc()
  
  def save_highscore(self, highscore):
    # Track when a new highscore is achieved
    highscore_file = str(self.ai_version) + self.sim_highscore_basename
    highscore_file = os.path.join(self.sim_data_dir, highscore_file)
    if not os.path.exists(self.sim_data_dir):
      os.makedirs(self.sim_data_dir)
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
    # Save the simulation model
    model_file = str(self.ai_version) + self.sim_model_basename
    model_file = os.path.join(self.sim_data_dir, model_file)
    if not os.path.exists(self.sim_data_dir):
      os.makedirs(self.sim_data_dir)
    self.model.save_model(self.trainer.optimizer, model_file)
    print(f"Saved simulation model ({model_file})")

  def save_sim_desc(self):
    # Save a descrion of the simulation model
    sim_desc_file = str(self.ai_version) + self.sim_desc_basename
    sim_desc_file = os.path.join(self.sim_data_dir, sim_desc_file)
    if not os.path.exists(self.sim_data_dir):
      os.makedirs(self.sim_data_dir)
    # Update the epsilon value
    self.set_config('epsilon_value', str(self.epsilon_value - self.n_games))
    self.set_config('nu_score', str(self.nu_score))
    self.set_config('nu_value', str(self.nu_value))
    self.set_config('num_games', str(self.n_games))
    self.set_config('highscore', str(self.highscore))
    with open(sim_desc_file, 'w') as config_file:
      self.config.write(config_file)
    print(f"Saved simulation description ({sim_desc_file})")

  def set_config(self, key, value):
    self.config['default'][key] = value

  def train_long_memory(self):
    if len(self.memory) > self.batch_size:
      # Sample a random batch of memories
      mini_sample = random.sample(self.memory, self.batch_size)
    else:
      mini_sample = self.memory

    # Get the states, actions, rewards, next_states, and dones from the mini_sample
    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.trainer.train_step(states, actions, rewards, next_states, dones)

  def train_short_memory(self, state, action, reward, next_state, done):
    self.trainer.train_step(state, action, reward, next_state, done)

  def get_action(self, state):
    # Random move: exploration vs exploitation...

    ## Epsilon - the more games played, the less likely to explore
    final_move = [0, 0, 0]
    epsilon = self.epsilon_value - self.n_games
    if random.randint(0, self.epsilon_value) < epsilon:
      # Random move based on the epsilon value
      final_move = [0, 0, 0]
      move = random.randint(0, 2)
      final_move[move] = 1
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      move = torch.argmax(prediction).item()
      final_move[move] = 1 

    ## Nu - Inject random moves
    nu = self.nu_value - self.nu_count
    if nu == 0:
      # nu has been depleted, re-initialize with higher nu score and nu value
      self.nu_num_games_same_score = 1
      self.nu_count = 0
      self.nu_score += 1
      self.initial_nu_value += 1
      # nu value increases by 2 if the scoreis 10+
      if self.highscore > 10:
        self.initial_nu_value += 1
      if self.initial_nu_value > 40:
        self.initial_nu_value = 40
      else:
        self.nu_value = self.initial_nu_value
      print(f"Nu depleted. Re-initializing with nu value {self.nu_value}, nu threshold score {self.nu_score}")
    
    if self.nu_num_games_same_score >= 30:
      print(f"AI played 30 games without improvement, forcing it back into learning mode")
      self.nu_num_games_same_score = 1
      if self.nu_score > 30:
        print(f"Reducing nu threshold score by 1")
        self.nu_score -= 1
      self.initial_nu_value = 40
      self.nu_value = 40

    nu_rand = random.randint(0, nu)
    if nu_rand < nu and self.game.score >= self.nu_score:
      self.nu_count += 1
      final_move = [0, 0, 0]
      move = random.randint(0, 2)
      final_move[move] = 1
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      move = torch.argmax(prediction).item()
      final_move[move] = 1 
 
    return final_move
