"""
AIAgentN.py

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
  """
  This AIAgent class is a variation of the AIAgent class.
  Instead of having a single model dedicated to covering
  the game during a range of scores i.e. 1-10, 11-20, etc.
  This class creates a model for each score above the starting
  point i.e. 3. 

  A single neural network would be dedicated to playing the
  game for every point e.g. for a score of 7. This would
  mean that the neural network would only be used to play
  the game for a score of 7.

  This class would also create a separate neural network
  for a score of 8, 9, 10 etc.

  The game environment would be less varied for each neural
  network. After the network has been trained for a while, the
  weights and biases of the neural network for a score of 5
  would be very different than the weights and biases for 
  network that handles the game at a score of 50, where 
  not coliding with yourself is a real challenge.

  However, the neural network dedicated to playing at a score
  of 5 would have a LOT more training steps than the network
  with a score of 50. That's because it would receive a training
  step for every game with a score above 5, wheras the network
  dedicated to playing at a score of 50 would only receive
  a training step for every game that reached a score of 50.

  To leverage the the training numbers that lower level
  neural networks receive, the weights and biases of the lower
  level neural networks would be transferred to the higher level.
  i.e. L4 would be copied to L5, L5 to L6 etc. In this ways
  the amount of training at lower levels would trickle up
  giving the higher levels a LOT more training steps than if 
  this hadn't happened.
  """
  def __init__(self, ini, log, game):
    self.ini = ini
    self.log = log
    self.game = game
    
    self.level = {}
    
    self.highscore = 0 # Current highscore
    self.cur_score = 0 # Current score is zero
    # Level 0, game score is zero, initialization
    self.add_level(0)

    # Nu and epsilon algorithms
    self.nu_algo = NuAlgo(ini, log)
    self.epsilon_algo = EpsilonAlgo(ini, log)

    self.game_num = 0 # Current game number
    self.prev_game_num = self.game_num - 1 # Previous game num
    self.save_highscore(0) # Initialize the highscore file

    # Used in the state map, this initializes it to some random direction
    self.last_dir = [ 0, 0, 1, 0 ]

    # Create the simulation data directory if it does not exist
    os.makedirs(self.ini.get('sim_data_dir'), exist_ok=True)

  def add_level(self, score):
    # Create an empty level
    self.level[score] = {
      'memory': ReplayMemory(self.ini),
      'model': LinearQNet(self.ini, self.log, score),
      'last_memory': [], # the last move we made and the state of things
      'num_games': 0, # How many games this level has played
    }

    model = self.level[score]['model']
    self.level[score]['trainer'] = QTrainer(self.ini, model, score)
        
    if score != 0:
      self.copy_previous_level(score)

  def copy_previous_level(self, score):
    # Backup the previous level's state
    tmp_file = str(self.ini.get('ai_version')) + '_tmp.pt'
    prev_trainer = self.level[score - 1]['trainer']
    prev_optimizer = prev_trainer.optimizer
    prev_model = self.level[score - 1]['model']
    prev_model.save_checkpoint(prev_optimizer, tmp_file)
    # Restore the previous level's state into the current state
    optimizer = self.level[score]['trainer'].optimizer
    model = self.level[score]['model']
    model.restore_model(optimizer, tmp_file)
    trainer = self.level[score]['trainer']
    # Preserve the total steps taken, as these were restored
    trainer.total_steps = prev_trainer.total_steps
    model.reset_steps()
    os.remove(tmp_file)

      
  def get_action(self, state):

    # Use the epsilon algorithm; exploration
    random_move = self.epsilon_algo.get_move()
    
    # Use the nu algorithm; exploration
    random_move = self.nu_algo.get_move(self.game.get_score())

    if random_move:
      # Random move was returned
      self.prev_game_num = self.game_num
      return random_move
    
    # AI agent based action
    final_move = [0, 0, 0]
    state0 = torch.tensor(state, dtype=torch.float)

    # Get the prediction
    score = self.game.get_score()
    prediction = self.level[score]['model'](state0)
    move = torch.argmax(prediction).item()
    final_move[move] = 1
    self.last_move = final_move
    return final_move

  def get_epsilon(self):
    return self.epsilon_algo
  
  def get_epsilon_value(self):
    return self.epsilon_algo.get_epsilon_value()

  def get_all_steps(self):
    all_steps = ''
    for model_num in self.level:
      all_steps = all_steps + 'Game# {:>5}, '.format(self.game_num)
      all_steps = all_steps + '{:>2}'.format(self.level[model_num]['model'].get_steps()) + ', '
      all_steps = all_steps + self.level[model_num]['trainer'].get_steps() + ', '
      all_steps = all_steps + self.level[model_num]['model'].get_total_steps() + ', '
      all_steps = all_steps + self.level[model_num]['trainer'].get_total_steps() + '\n'
    return all_steps
  
  def get_model_steps(self, score):
    return self.level[score]['model'].get_steps()

  def get_nu_algo(self):
    return self.nu_algo
  
  def get_binary(self, num):
    # Get the length of the snake in binary.
    # This is used in the state map, the get_state() function.
    bin_str = format(int(num), 'b')
    out_list = []
    for bit in range(len(bin_str)):
      out_list.append(bin_str[bit])
    for zero in range(7 - len(out_list)):
      out_list.insert(0, '0')
    for x in range(7):
      out_list[x] = int(out_list[x])
    return out_list
  
  def get_num_games(self, level):
    return 'L{:>2} games# {:>4}'.format(level, self.level[level]['num_games'])

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

    # Get the snake length in binary
    slb = self.get_binary(len(game.snake))

    # Get the last direction
    ldir = self.last_dir

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
    for aDir in self.last_dir:
      state.append(aDir)
    self.last_dir = [ dir_l, dir_r, dir_u, dir_d ]
    return np.array(state, dtype=int)
  
  def get_trainer_steps(self, score):
    return self.level[score]['trainer'].get_steps()

  def increment_games(self):
    self.game_num += 1

  def played_game(self, score):
    self.level[score]['num_games'] += 1
    self.epsilon_algo.played_game()
    self.nu_algo.played_game(score)
    if score and score >= self.cur_score:
      self.copy_previous_level(score)
    self.cur_score = score

  def print_model(self):
    # All of the models share the same structure, return one of them
    print(self.level[0]['model'])

  def remember(self, state, action, reward, next_state, done):
    # Store the state, action, reward, next_state, and done in memory
    # Recall that memory is a deque, so it will automatically remove the oldest memory 
    # if the memory exceeds MAX_MEMORY
    score = self.game.get_score()
    self.level[score]['memory'].append((state, action, reward, next_state, done))   
    # Reward lower levels if the reward was 10, indicating that we got some food.
    if reward == self.ini.get('reward_food'):
      self.reward_lower_levels()
    # Train the level below us at the same time
    if score > 0:      
      self.train_lower_level(state, action, reward, next_state, done)

  def reset_epsilon_injected(self):
    self.epsilon_algo.reset_injected()
  def reset_nu_algo_injected(self):
    self.nu_algo.reset_injected()

  def reset_model_steps(self):
    for model_num in self.level:
      self.level[model_num]['model'].reset_steps()

  def reset_trainer_steps(self):
    for model_num in self.level:
      self.level[model_num]['trainer'].reset_steps()

  def reward_lower_levels(self):
    # When the current network decision resulted in a reward it was also
    # due to the last move made by all the lower score AIs. Reward them too
    # based on their last move. Retrieve their last memory, alter it and 
    # feed it back to them as if it was a normal memory.
    cur_score = self.cur_score
    while cur_score != 0:
      # Get the previous level's last memory, [state, action, reward, next_state, done]
      prev_memory = self.level[cur_score - 1]['memory'].pop()
      # The get_last memory returns a list in the form
      # [state, action, reward, next_state, done]. 
      food_reward = float(self.ini.get('reward_food')) / 10
      #print(f"prev_memory: ", prev_memory)
      #print(f"prev_memory[2] {prev_memory[2]}")
      #print(f"type(prev_memory[2]) {type(prev_memory[2])}")
      self.level[cur_score - 1]['trainer'].train_step(prev_memory[0], prev_memory[1], food_reward, prev_memory[3], prev_memory[4])
      self.level[cur_score - 1]['memory'].append((prev_memory[0], prev_memory[1], food_reward, prev_memory[3], prev_memory[4]))
      cur_score -= 1
  

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
        file_handle.write(str(self.game_num) + ',' + str(highscore) + "\n")

  def set_highscore(self, score):
    # We achieved a new highscore. 
    self.highscore = score
    self.cur_score = score
    self.save_highscore(score)
    # Reward all the previous neural networks for their last move that got
    # us here
    self.reward_lower_levels()

  def set_nu_algo_highscore(self, score):
    # We need to let *all* of the NuAlgo instances know about the new high score
    self.nu_algo.new_highscore(score)
    
  def train_long_memory(self):
    score = self.game.get_score()
    # Get the states, actions, rewards, next_states, and dones from the mini_sample
    mini_sample = self.level[score]['memory'].get_memory()
    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.level[score]['trainer'].train_step(states, actions, rewards, next_states, dones)

  def train_lower_level(self, state, action, reward, next_state, done):
    # Lower levels are regularly promoted to higher levels. So we'll
    # Save this experience into the lower level, so when it's promoted
    # it will have the added benefit of this training experience.
    if self.cur_score > 0:
      self.level[self.cur_score - 1]['memory'].pop()
      self.level[self.cur_score - 1]['trainer'].train_step(state, action, reward, next_state, done)
      self.level[self.cur_score - 1]['memory'].append((state, action, reward, next_state, done))

  def train_short_memory(self, state, action, reward, next_state, done):
    score = self.game.get_score()
    # Make sure we have a network to handle this score
    if score not in self.level:
      self.add_level(score)
    # Execute the training step
    self.level[score]['trainer'].train_step(state, action, reward, next_state, done)

    

      
