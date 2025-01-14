import torch
import random
import numpy as np
import os
from collections import deque
from time import time
from ai_snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot, plot2
import matplotlib.pyplot as plt


random.seed(42)
torch.manual_seed(1970)

MAX_MEMORY = 100_000 # Maximum memory for the replay buffer
BATCH_SIZE = 1000 # Batch size for the replay buffer
# Number of nodes in the input layer i.e. the number of nodes used to 
# describe the state of the game
INPUT_NODES = 40

# Include previous directioni in the state map.
# Only enable in the "if AI_VERSION == <ver>:" block as this changes
# the shape. DO NOT CHANGE HERE.
ENABLE_HISTORY = True
ENABLE_LONG_HISTORY = False

# Enable the adding nn.ReLU() layers before the linear layers
ENABLE_RELU = True

## Block 1: There *MUST* be at least 1 layer
B1_LAYERS = 1
# Number of nodes in block 1 layer(s)
B1_NODES = 512 


## Block 2: Optional additional block (with different number of nodes than B1)
# Number of nodes in block 2 layer(s)
B2_NODES = 1024
# Number of block 2 layers
B2_LAYERS = 0 # Can be zero

## Block 3: Optional addition block. 
# If you want to use36 this, then you must have at least 1 block 2 layer
# Number of nodes in block 3 layer(s)
B3_NODES = 512
# Number of block 3 layers
B3_LAYERS = 0

# Number of nodes in the output layer. This corresponds to valid moves
# that the snake can make i.e. left, right or continue straight
# You *CANNOT* change this.
OUTPUT_NODES = 3

# Discount rate, must be smaller than 1  
DISCOUNT = 0.8 
# Learning rate
LR = 0.001
# Epsilon value, for exploration. While the number of games is less than
# this value then there is a random chance that there will be a random snake
# i.e. not decided by the AI
EPSILON_VALUE = 150
ENABLE_EPSILON = True

# Random move value
RANDOM_MOVE_VALUE = 20
ENABLE_RANDOM_MOVES = False

## Simulation checkpoint save info
SIM_CHECKPOINT_DIR = './models'
SIM_CHECKPOINT_FILE = 'ai_checkpoint_v'
SIM_CHECKPOINT_FILE_SUFFIX = '.ptc'
SIM_CHECKPOINT_FREQ = 50

# Default is to disable the "look ahead" feature
BAD_MOVE_LIMIT = 0

# The version of this codebase. This is used to allow me to have code branching and
# model changes depending on the version of the code base. This allows me to easily
# revert back or select specific versions of the AI code to be run.
AI_VERSION = 142

if AI_VERSION == 147:
  B1_NODES = 300
  B1_LAYERS = 5
  EPSILON_VALUE = 2000

# LR = 0.0005 is TERRIBLE!!!
# LR = 0.005 is also TERRIBLE!!!

# Restarted after 1800 to see data more clearly
if AI_VERSION == 142:
  B1_NODES = 300
  B1_LAYERS = 1
  B2_NODES = 500
  B2_LAYERS = 1
  B3_NODES = 700
  B3_LAYERS = 2
  EPSILON_VALUE = 0

### Reference version
### It started to really take off at game 150
#if AI_VERSION == 145:
#  B1_NODES = 300
#  B1_LAYERS = 4
#  EPSILON_VALUE = 100

### Reference version #1
### This configuration achieves the circling solution around game 300, but this strategy
### limits the top score to about 50, with luck. 
#if AI_VERSION == 133:
#  B1_NODES = 300
#  B1_LAYERS = 1
#  EPSILON_VALUE = 0

# Enable historical moves
if not ENABLE_HISTORY:
  INPUT_NODES = 36

if ENABLE_LONG_HISTORY:
  INPUT_NODES = 52

class Agent:
  def __init__(self, game):
    self.game = game
    self.n_games = 0 # Number of games played
    self.epsilon = 0 # Epsilon value, for exploration (i.e. vs exploitation)
    self.gamma = DISCOUNT # Discount rate, for future rewards
    # If memory exceeds MAX_MEMORY, oldest memory is removed i.e. popleft()
    self.memory = deque(maxlen=MAX_MEMORY) 
    self.model = Linear_QNet(INPUT_NODES, 
                             B1_NODES, B1_LAYERS, 
                             B2_NODES, B2_LAYERS, 
                             B3_NODES, B3_LAYERS, 
                             OUTPUT_NODES, 
                             ENABLE_RELU, AI_VERSION)
    self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    self.last_dirs = [ 0, 0, 1, 0 ]
    self.second_last_dirs = [ 0, 0, 1, 0 ]
    self.third_last_dirs = [ 0, 0, 1, 0 ]
    self.random_move_count = 0
    # Load the simulation state from file if it exists
    self.load_checkpoint()
    print(f"Epsilon value is {EPSILON_VALUE}")

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
      (dir_r and game.is_self_collision(point_r)),
      (dir_l and game.is_self_collision(point_l)),
      (dir_u and game.is_self_collision(point_u)),
      (dir_d and game.is_self_collision(point_d)),

      # Danger right
      (dir_u and game.is_self_collision(point_r)),
      (dir_d and game.is_self_collision(point_l)),
      (dir_l and game.is_self_collision(point_u)),
      (dir_r and game.is_self_collision(point_d)),

      # Danger left
      (dir_d and game.is_self_collision(point_r)) or
      (dir_u and game.is_self_collision(point_l)) or
      (dir_r and game.is_self_collision(point_u)) or
      (dir_l and game.is_self_collision(point_d)),

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
    if ENABLE_HISTORY:
      for aDir in self.last_dirs:
        state.append(aDir)
      self.last_dirs = [ dir_l, dir_r, dir_u, dir_d ] 

    if ENABLE_LONG_HISTORY:
      for aDir in self.third_last_dirs:
        state.append(aDir)
      for aDir in self.second_last_dirs:
        state.append(aDir)
      for aDir in self.last_dirs:
        state.append(aDir)
      self.third_last_dirs = self.second_last_dirs
      self.second_last_dirs = self.last_dirs
      self.last_dirs = [ dir_l, dir_r, dir_u, dir_d ] 

    return np.array(state, dtype=int)

  def remember(self, state, action, reward, next_state, done):
    # Store the state, action, reward, next_state, and done in memory
    # Recall that memory is a deque, so it will automatically remove the oldest memory 
    # if the memory exceeds MAX_MEMORY
    self.memory.append((state, action, reward, next_state, done))

  def save_checkpoint(self):
    # Save the simulation state
    checkpoint_file = SIM_CHECKPOINT_FILE + str(AI_VERSION) + SIM_CHECKPOINT_FILE_SUFFIX
    checkpoint_file = os.path.join(SIM_CHECKPOINT_DIR, checkpoint_file)
    if not os.path.exists(SIM_CHECKPOINT_DIR):
      os.makedirs(SIM_CHECKPOINT_DIR)
    self.model.save_checkpoint(self.trainer.optimizer, checkpoint_file, self.game.num_games)
    print(f"Saved simulation checkpoint ({checkpoint_file})")

  def load_checkpoint(self):
    checkpoint_file = SIM_CHECKPOINT_FILE + str(AI_VERSION) + SIM_CHECKPOINT_FILE_SUFFIX
    checkpoint_file = os.path.join(SIM_CHECKPOINT_DIR, checkpoint_file)
    if os.path.isfile(checkpoint_file):
      optimizer = self.trainer.optimizer
      self.model.load_checkpoint(optimizer, checkpoint_file)
      print(f"Loaded simulation checkpoint ({checkpoint_file})")

  def train_long_memory(self):
    if len(self.memory) > BATCH_SIZE:
      # Sample a random batch of memories
      mini_sample = random.sample(self.memory, BATCH_SIZE)
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
    self.epsilon = EPSILON_VALUE - self.n_games
    if ENABLE_EPSILON and \
      random.randint(0, EPSILON_VALUE) < self.epsilon:
      # Random move based on the epsilon value
      final_move = [0, 0, 0]
      move = random.randint(0, 2)
      final_move[move] = 1
    elif ENABLE_RANDOM_MOVES and \
      (self.game.frame_iteration % RANDOM_MOVE_VALUE) == 0:
      # Random move
      self.random_move_count += 1
      final_move = [0, 0, 0]
      move = random.randint(0, 2)
      final_move[move] = 1
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      move = torch.argmax(prediction).item()
      final_move[move] = 1 
    return final_move

def train(game):
  plot_scores = [] # Scores for each game
  plot_mean_scores = [] # Average scores over a rolling window
  plot_times = [] # Times for each game
  plot_mean_times = [] # Average times over a rolling window
  total_score = 0 # Score for the current game
  record = 0 # Best score
  agent = Agent(game)
  game.set_agent(agent)
  game.set_sim_checkpoint_freq(SIM_CHECKPOINT_FREQ)
  fig = plt.figure(figsize=(12,4), layout="tight")
  spec = fig.add_gridspec(ncols=1, nrows=2)
      
  while True:
    # Get old state
    state_old = agent.get_state()
    # Get move
    final_move = agent.get_action(state_old)
    # Perform move and get new state
    reward, done, score = game.play_step(final_move)
    state_new = agent.get_state()
    # Train short memory
    agent.train_short_memory(state_old, final_move, reward, state_new, done)
    # Remember
    agent.remember(state_old, final_move, reward, state_new, done)
    if done:
      # Train long memory
      game.reset()
      
      agent.n_games += 1
      agent.train_long_memory()
      if score > record:
        record = score
        agent.save_checkpoint()
        game.sim_high_score = record

      print('Snake AI (v' + str(AI_VERSION) + ') ' + \
            'Game' + '{:>4}'.format(agent.n_games) + ', ' + \
            'Score' + '{:>4}'.format(score) + ', ' + \
            'Record' + '{:>4}'.format(record) + ', ' + \
            'Time ' + '{:>7}'.format(game.elapsed_time) + 's' + \
            ' - ' + game.lose_reason)

      plot_scores.append(score)
      total_score += score
      mean_score = round(total_score / agent.n_games, 1)
      plot_mean_scores.append(mean_score)
      plot_times.append(game.elapsed_time)
      mean_time = round(game.sim_time / agent.n_games, 1)
      plot_mean_times.append(mean_time)
      plot(plot_scores, plot_mean_scores, plot_times, plot_mean_times, AI_VERSION)
      #plot2(fig, spec, plot_scores, plot_mean_scores, plot_times, plot_mean_times, AI_VERSION)
  

if __name__ == '__main__':
  game = SnakeGameAI(AI_VERSION)
  train(game)

