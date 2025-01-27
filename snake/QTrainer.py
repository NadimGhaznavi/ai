"""
QTrainer.py

This file contains the QTrainer class.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os, sys

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig


class QTrainer:
  def __init__(self, ini, model, model_level):
    """
    The constructor accepts the following parameters:
        * model - A sub-class of nn.Module
        * lr    - The learning rate
        * gamma - The gamma value
    """
    self.model_level = model_level
    torch.manual_seed(ini.get('random_seed'))
    self.lr = ini.get('learning_rate')
    self.gamma = ini.get('discount')
    self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
    # Mean Squared Error Loss... 
    self.criterion = nn.MSELoss()
    self.model = model
    # Keep track of the number of steps executed by this instance
    self.steps = 0
  
  def get_steps(self):
    return 'L{} trainer steps# {:>5}'.format(self.model_level, self.steps)
  
  def reset_steps(self):
    """
    Resets the number of steps to zero.
    """
    self.steps = 0
                
  def train_step(self, state, action, reward, next_state, game_over):
    """
    The train_step() function accepts the following parameters
        * state
        * action
        * reward,
        * next_state
        * game_over
        * cur_score
    """
    state = torch.tensor(np.array(state), dtype=torch.float)
    next_state = torch.tensor(np.array(next_state), dtype=torch.float)
    action = torch.tensor(action, dtype=torch.long)
    reward = torch.tensor(reward, dtype=torch.float)
    
    if len(state.shape) == 1:
      # Add a batch dimension
      state = torch.unsqueeze(state, 0)
      next_state = torch.unsqueeze(next_state, 0)
      action = torch.unsqueeze(action, 0)
      reward = torch.unsqueeze(reward, 0)
      game_over = (game_over, )
      
    # 1. predicted Q values with current state
    pred = self.model(state)

    # 2. Q_new = r + y * max(next_predicted Q value) -> only do this if not done
    target = pred.clone()
    for idx in range(len(game_over)):
      # Track the number of steps executed by this instance
      self.steps += 1
      Q_new = reward[idx]
      if not game_over[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
      target[idx][torch.argmax(action).item()] = Q_new

    self.optimizer.zero_grad() # Reset the gradients to zero
    loss = self.criterion(target, pred) # Calculate the loss
    #print("Loss: ", loss.item())
    loss.backward() # Backpropagate the loss
    self.optimizer.step() # Adjust the weights
