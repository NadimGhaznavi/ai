"""
AIAgentR.py

This class contains the AIAgent class. An instance of this class acts as
the player to the AISnakeGame. It's based on the AIAgent code.
"""
import os, sys
from collections import deque
from collections import namedtuple
import numpy as np
import torch

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from TrainerR import TrainerR
from ModelR import ModelR
from SnakeGameElement import Direction
from SnakeGameElement import Point
from EpsilonAlgo import EpsilonAlgo
from ReplayMemory import ReplayMemory
from NuAlgo import NuAlgo

class AIAgentR:
  def __init__(self, ini, log, game):
    self.ini = ini
    self.log = log
    self.game = game
    
    level_score = self.ini.get('level_score')

    # Level 1 initialization
    self.memory = ReplayMemory(ini)
    self.model = ModelR(ini, log, level_score)
    self.trainer = TrainerR(ini, self.model, level_score)

    self.nu_algo = NuAlgo(ini, log)
    self.epsilon_algo = EpsilonAlgo(ini, log)

    # Used in the state map, this initializes it to some random direction
    self.last_dirs = [ 0, 0, 1, 0 ]
    self.highscore = 0 # Current highscore
    self.batch_size = self.ini.get('batch_size')
    self.scores = {}

    self.n_games = 0 # Number of games played
    self.n_games_buf = -1    
    
    self.save_highscore(0) # Initialize the highscore file

    self.loss = {} # Store losses here
    self.cur_loss = None

    # Create the simulation data directory if it does not exist
    os.makedirs(self.ini.get('sim_data_dir'), exist_ok=True)

  def get_action(self, state):

    # Use the epsilon algorithm; exploration
    random_move = self.epsilon_algo.get_move()
    
    # Use the nu algorithm; exploration
    if not random_move:
      random_move = self.nu_algo.get_move(self.game.get_score())

    if random_move:
      # Random move was returned
      self.n_games_buf = self.n_games
      return random_move
    
    # AI agent based action
    final_move = [0, 0, 0]
    state0 = torch.tensor(state, dtype=torch.float)

    # Get the prediction
    print("DEBUG state0: ", state0)
    prediction = self.model(state0)
    print(f"DEBUG prediction: {prediction}")
    print(f"DEBUG prediction.shape: {prediction.shape}")
 
    move = torch.argmax(prediction).item()
    print(f"DEBUG: move: {move}")
    final_move[move] = 1 
    return final_move

  def get_avg_loss(self, score):
    # self.loss is a dictionary of lists, where the key is the
    # score, and the value is a list of losses for that score.
    # This function returns the average loss for a particular score.
    scores = self.loss[score]
    total = 0
    for a_score in scores:
      total += a_score
    return total / len(scores)
  
  def get_highscore(self):
    return self.highscore
  
  def get_epsilon(self):
    return self.epsilon_algo
  
  def get_epsilon_value(self):
    return self.epsilon_algo.get_epsilon_value()

  def get_model_steps(self, score):
    return self.model.get_steps()

  def get_nu_algo(self):
    return self.nu_algo
  
  def get_num_games(self):
    return self.n_games
  
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
    return self.game.get_state()
  
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
    return self.trainer.get_steps()

  def increment_games(self):
    self.n_games += 1

  def log_loss(self):
    loss_str = 'AIAgent          Loss : '
    count = 0
    for level in sorted(self.loss.keys()):
      loss = self.get_avg_loss(level)
      if count and count % 10 == 0:
        self.log.log(loss_str)
        loss_str = 'AIAgent     Loss + {:<2} : '.format(count)
      loss_str = loss_str + '{:>8.2f}'.format(round(loss,2))
      count += 1
    self.log.log(loss_str)

  def log_scores(self):
    total_games = self.get_num_games()
    # Fill in any blanks
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
    self.log.log('AIAgent (v' + str(self.ini.get('ai_version')) + \
                 ')  Game :{:>5}'.format(total_games) + \
                  ', Score: {:>2}'.format(self.game.get_score()) + \
                    ', Highscore: {:>2}'.format(self.get_highscore()) + \
                      ', Loss: {:>6.2f}'.format(self.trainer.get_cur_loss()))
                        
    self.log.log('AIAgent        Scores : ' + \
                 '{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}'.format(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0))
    self.log.log('AIAgent               : ' + ('-' * 80))
    score_str = 'AIAgent         Count : '
    count = 0
    for score in self.scores.keys():
      if count and count % 10 == 0:
        self.log.log(score_str) # Chop off last ', ' 
        score_str = 'AIAgent    Count + {:<2} : '.format(count)
      score_str = score_str + '{:>8}'.format(self.scores[score])
      count += 1
    self.log.log(score_str) 
     
  
  def played_game(self, score):
    self.epsilon_algo.played_game()
    self.nu_algo.played_game(score)
    if score in self.scores:
      self.scores[score] += 1
    else:
      self.scores[score] = 1

  def print_model(self):
    # All of the models share the same structure, return one of them
    print(self.model)

  def remember(self, state, action, reward, next_state, done):
    # Store the state, action, reward, next_state, and done in memory
    # Recall that memory is a deque, so it will automatically remove the oldest memory 
    # if the memory exceeds MAX_MEMORY
    self.memory.append((state, action, reward, next_state, done))

  def reset_epsilon_injected(self):
    self.epsilon_algo.reset_injected()
  def reset_nu_algo_injected(self):
    self.nu_algo.reset_injected()

  def reset_model_steps(self):
    self.model.reset_steps()

  def reset_trainer_steps(self):
    self.trainer.reset_steps()

  def save_checkpoint(self):
    # Save the models for each level
    if self.ini.get('sim_checkpoint_enable'):
      data_dir = self.ini.get('sim_data_dir')
      checkpoint_basename = self.ini.get('sim_checkpoint_basename')
      checkpoint_file = os.path.join(data_dir, str(self.ini.get('ai_version')) + checkpoint_basename)
      optimizer = self.trainer.optimizer
      self.model.save_checkpoint(optimizer, checkpoint_file)
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
    # Get the states, actions, rewards, next_states, and dones from the mini_sample
    mini_sample = self.memory.get_memory()
    states, actions, rewards, next_states, dones = zip(*mini_sample)

  def train_short_memory(self, state, action, reward, next_state, done):
    self.trainer.train_step(state, action, reward, next_state, done)
    loss = self.trainer.get_cur_loss()
    self.cur_loss = loss
    if self.game.get_score() not in self.loss:
      # Store the loss for the current score in a fixed length array (deque)
      self.loss[self.game.get_score()] = deque(maxlen=1000)
    self.loss[self.game.get_score()].append(loss)