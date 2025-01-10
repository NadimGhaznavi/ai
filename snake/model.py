import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

MODEL_DIR = './model'
MODEL_FILE = 'ai_model_v'
MODEL_FILE_SUFFIX = 'pth'

class Linear_QNet(nn.Module):
  def __init__(self, input_nodes, hidden_nodes, hidden_layers, output_nodes, ai_version):
    super().__init__()
    self.ai_version = ai_version
    self.layer_stack = nn.Sequential()
    self.layer_stack.append(nn.Linear(in_features=input_nodes, out_features=hidden_nodes))
    self.layer_stack.append(nn.ReLU())
    hidden_layer_count = 0
    while hidden_layer_count != hidden_layers:
      self.layer_stack.append(nn.Linear(in_features=hidden_nodes,
                                        out_features=hidden_nodes))
      self.layer_stack.append(nn.ReLU())
      hidden_layer_count += 1
    self.layer_stack.append(nn.Linear(in_features=hidden_nodes,
                out_features=output_nodes))
  
  def forward(self, x):
    return self.layer_stack(x)
  
  def save(self):
    file_name = MODEL_FILE + str(self.ai_version) + '.' + MODEL_FILE_SUFFIX
    if not os.path.exists(MODEL_DIR):
      os.makedirs(MODEL_DIR)
    file_name = os.path.join(MODEL_DIR, file_name)
    torch.save(self.state_dict(), file_name)

  def load(self):
    file_name = MODEL_FILE + str(self.ai_version) + '.' + MODEL_FILE_SUFFIX
    file_name = os.path.join(MODEL_DIR, file_name)
    if os.path.isfile(file_name):
      self.load_state_dict(torch.load(file_name, weights_only=False))
    
class QTrainer:
  def __init__(self, model, lr, gamma):
    self.lr = lr
    self.gamma = gamma
    self.model = model
    self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
    # Mean Squared Error Loss... 
    self.criterion = nn.MSELoss()
    
  def train_step(self, state, action, reward, next_state, game_over):
    np_state = np.array(state)
    state = torch.tensor(np_state, dtype=torch.float)
    next_state = torch.tensor(next_state, dtype=torch.float)
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
      Q_new = reward[idx]
      if not game_over[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
      target[idx][torch.argmax(action).item()] = Q_new
      
    self.optimizer.zero_grad() # Reset the gradients to zero
    loss = self.criterion(target, pred) # Calculate the loss
    loss.backward() # Backpropagate the loss
    self.optimizer.step() # Adjust the weights