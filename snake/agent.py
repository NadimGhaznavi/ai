import torch
import random
import numpy as np
from collections import deque
from ai_snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
from time import time

random.seed(42)
torch.manual_seed(1970)

MAX_MEMORY = 100_000 # Maximum memory for the replay buffer
BATCH_SIZE = 1000 # Batch size for the replay buffer
HIDDEN_SIZE = 32 # Hidden size for the model
DISCOUNT = 0.8 # Discount rate, must be smaller than 1  
LR = 0.003 # Learning rate
EPSILON_VALUE = 200 # Epsilon value, for exploration (i.e. vs exploitation)
MODEL_VERSION = 9

if MODEL_VERSION > 0:
  LR = 0.001
  HIDDEN_SIZE = 256
  EPSION_VALUE = 100

if MODEL_VERSION > 7:
  EPSILON_VALUE = 150
  HIDDEN_SIZE = 64

if MODEL_VERSION == 9:
  LR = 0.001
  EPSION_VALUE = 200
  HIDDEN_SIZE = 128

class Agent:

  def __init__(self):
    self.n_games = 0 # Number of games played
    self.epsilon = 0 # Epsilon value, for exploration (i.e. vs exploitation)
    self.gamma = DISCOUNT # Discount rate, for future rewards
    # If memory exceeds MAX_MEMORY, oldest memory is removed i.e. popleft()
    self.memory = deque(maxlen=MAX_MEMORY) 
    self.model = Linear_QNet(11, HIDDEN_SIZE, 3, MODEL_VERSION) 
    self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

  def get_state(self, game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)

    point_l2 = Point(head.x - 40, head.y)
    point_r2 = Point(head.x + 40, head.y)
    point_u2 = Point(head.x, head.y - 40)
    point_d2 = Point(head.x, head.y + 40)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
      # Danger straight
      (dir_r and game.is_collision(point_r)) or
      (dir_l and game.is_collision(point_l)) or
      (dir_u and game.is_collision(point_u)) or
      (dir_d and game.is_collision(point_d)),

      # Danger right
      (dir_u and game.is_collision(point_r)) or
      (dir_d and game.is_collision(point_l)) or
      (dir_l and game.is_collision(point_u)) or
      (dir_r and game.is_collision(point_d)),

      # Danger left
      (dir_d and game.is_collision(point_r)) or
      (dir_u and game.is_collision(point_l)) or
      (dir_r and game.is_collision(point_u)) or
      (dir_l and game.is_collision(point_d)),

      # Move direction
      dir_l,
      dir_r,
      dir_u,
      dir_d,

      # Food location
      game.food.x < game.head.x, # Food left
      game.food.x > game.head.x, # Food right
      game.food.y < game.head.y, # Food up
      game.food.y > game.head.y, # Food down
    ]

    return np.array(state, dtype=int)

  def remember(self, state, action, reward, next_state, done):
    # Store the state, action, reward, next_state, and done in memory
    # Recall that memory is a deque, so it will automatically remove the oldest memory 
    # if the memory exceeds MAX_MEMORY
    self.memory.append((state, action, reward, next_state, done))

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
    # The more games played, the less likely to explore i.e. 
    self.epsilon = EPSILON_VALUE - self.n_games
    #self.epsilon = 0
    final_move = [0, 0, 0]
    if random.randint(0, EPSILON_VALUE) < self.epsilon:
      # Random move
      move = random.randint(0, 2)
      final_move[move] = 1
    else:
      state0 = torch.tensor(state, dtype=torch.float)
      prediction = self.model(state0)
      move = torch.argmax(prediction).item()
      final_move[move] = 1
    return final_move

def train():
  plot_scores = [] # Scores for each game
  plot_mean_scores = [] # Average scores over a rolling window
  plot_game_times = []
  total_score = 0 # Score for the current game
  record = 0 # Best score
  agent = Agent()
  game = SnakeGameAI(MODEL_VERSION)
  while True:
    start_time = time()
    # Get old state
    state_old = agent.get_state(game)
    # Get move
    final_move = agent.get_action(state_old)
    # Perform move and get new state
    reward, done, score = game.play_step(final_move)
    state_new = agent.get_state(game)
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
        agent.model.save()

      # Calculate elapsed game time
      end_time = time()
      total_time = round((end_time - start_time), 1)
      plot_game_times.append(total_time)

      if total_time > 60:
        min = str(total_time / 60)
        sec = str(total_time % 60)
        total_time = min + ' min ' + sec

      game_str = 'Game v' + str(MODEL_VERSION)
      print('Game (v' + str(MODEL_VERSION) + '){:>4}'.format(agent.n_games) + ', ' + \
            'Score' + '{:>4}'.format(score) + ', ' + \
            'Record' + '{:>4}'.format(record) + ', ' + \
            'Game Time ' + str(total_time) +  ' sec')

      plot_scores.append(score)
      total_score += score
      mean_score = total_score / agent.n_games
      plot_mean_scores.append(mean_score)
      plot(plot_scores, plot_mean_scores, plot_game_times, MODEL_VERSION)

if __name__ == '__main__':
  train()

