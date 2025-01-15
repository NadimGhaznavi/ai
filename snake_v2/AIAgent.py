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

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig
from QTrainer import QTrainer
from SnakeGameElement import Direction
from SnakeGameElement import Point

class AIAgent:
  def __init__(self, game, model, ai_version):
    ini = AISnakeGameConfig()
    self.ai_version = ai_version
    self.batch_size = ini.batch_size()
    self.epsilon_value = ini.epsilon_value() # Epsilon value, for exploration (i.e. vs exploitation)
    self.gamma = ini.discount() # Discount rate, for future rewards
    self.game = game
    self.last_dirs = [ 0, 0, 1, 0 ]
    self.learning_rate = ini.learning_rate()
    self.memory = deque(maxlen=ini.max_memory())
    self.model = model
    self.n_games = 0 # Number of games played
    self.random_move_count = 0
    self.sim_checkpoint_basename = ini.sim_checkpoint_basename()
    self.sim_checkpoint_dir = ini.sim_checkpoint_dir()
    self.sim_checkpoint_file_suffix = ini.sim_checkpoint_file_suffix()
    self.sim_model_basename = ini.sim_model_basename()
    self.sim_model_desc_basename = ini.sim_model_desc_basename()
    self.sim_model_file_file_suffix = ini.sim_model_file_suffix()
    self.sim_model_dir = ini.sim_model_dir()
    self.trainer = QTrainer(self.model)

    self.load_checkpoint() # Load the simulation state from file if it exists
    print(f"Epsilon value is {self.epsilon_value}")

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
    checkpoint_file = self.sim_checkpoint_basename + str(self.ai_version) + '.' + self.sim_checkpoint_file_suffix
    checkpoint_file = os.path.join(self.sim_checkpoint_dir, checkpoint_file)
    if os.path.isfile(checkpoint_file):
      optimizer = self.trainer.optimizer
      self.model.load_checkpoint(optimizer, checkpoint_file)
      #self.n_games = self.model['num_games']
      print(f"Loaded simulation checkpoint ({checkpoint_file})")

  def load_model(self):
    model_file = self.sim_model_basename + str(self.ai_version) + '.' + self.sim_model_file_file_suffix
    model_file = os.path.join(self.sim_model_dir, model_file)
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
    checkpoint_file = self.sim_checkpoint_basename + str(self.ai_version) + '.' + self.sim_checkpoint_file_suffix
    checkpoint_file = os.path.join(self.sim_checkpoint_dir, checkpoint_file)
    if not os.path.exists(self.sim_checkpoint_dir):
      os.makedirs(self.sim_checkpoint_dir)
    self.model.save_checkpoint(self.trainer.optimizer, checkpoint_file, self.game.num_games)
    print(f"Saved simulation checkpoint ({checkpoint_file})")
  
  def save_model(self):
    # Save the simulation model
    model_file = self.sim_model_basename + str(self.ai_version) + '.' + self.sim_model_file_file_suffix
    model_file = os.path.join(self.sim_model_dir, model_file)
    if not os.path.exists(self.sim_model_dir):
      os.makedirs(self.sim_model_dir)
    self.model.save_model(self.trainer.optimizer, model_file)
    print(f"Saved simulation model ({model_file})")

  def save_model_desc(self, in_features,
                      b1n, b1l, b2n, b2l, b3n, b3l,
                      out_features,
                      enable_relu, ai_version):
    # Save a descrion of the simulation model
    model_desc_file = self.sim_model_desc_basename + str(ai_version) + '.txt'
    model_desc_file = os.path.join(self.sim_model_dir, model_desc_file)
    if not os.path.exists(self.sim_model_dir):
      os.makedirs(self.sim_model_dir)
    with open(model_desc_file, 'w') as file_handle:
      file_handle.write("[default]\n")
      file_handle.write("in_features = " + str(in_features) + "\n")
      file_handle.write("b1n = " + str(b1n) + "\n")
      file_handle.write("b1l = " + str(b1l) + "\n")
      file_handle.write("b2n = " + str(b2n) + "\n")
      file_handle.write("b2l = " + str(b2l) + "\n")
      file_handle.write("b3n = " + str(b3n) + "\n")
      file_handle.write("b3l = " + str(b3l) + "\n")
      file_handle.write("out_features = " + str(out_features) + "\n")
      file_handle.write("enable_relu = " + str(enable_relu) + "\n")
      file_handle.close()
    print(f"Saved simulation model description ({model_desc_file})")

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
    # The more games played, the less likely to explore
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
    return final_move
