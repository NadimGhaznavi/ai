import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, model_version):
    super().__init__()
    self.model_version = model_version
    self.linear_in = nn.Linear(input_size, hidden_size)
    if self.model_version > 3:
      self.linear_hidden_1 = nn.Linear(hidden_size, hidden_size)
    if self.model_version > 4:
      self.linear_hidden_2 = nn.Linear(hidden_size, hidden_size)
    #self.linear_hidden_3 = nn.Linear(hidden_size, hidden_size)
    #self.linear_hidden_4 = nn.Linear(hidden_size, hidden_size)
    #self.linear_hidden_5 = nn.Linear(hidden_size, hidden_size)
    #self.linear_hidden_6 = nn.Linear(hidden_size, hidden_size)
    #self.linear_hidden_7 = nn.Linear(hidden_size, hidden_size)
    self.linear_out = nn.Linear(hidden_size, output_size)
    
  
  def forward(self, x):
    x = F.relu(self.linear_in(x))
    if self.model_version > 3:
      x = self.linear_hidden_1(x)
    if self.model_version > 4:
      x = self.linear_hidden_2(x)
    x = self.linear_out(x)
    return x
  
  def save(self, file_name='model.pth'):
    model_folder_path = './model'
    if not os.path.exists(model_folder_path):
      os.makedirs(model_folder_path)
    file_name = os.path.join(model_folder_path, file_name)
    torch.save(self.state_dict(), file_name)

  def load(self, file_name='model.pth'):
    file_name = os.path.join('./model', file_name)
    self.load_state_dict(torch.load(file_name))
    
class QTrainer:
  def __init__(self, model, lr, gamma):
    self.lr = lr
    self.gamma = gamma
    self.model = model
    self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
    # Mean Squared Error Loss... 
    self.criterion = nn.MSELoss()
    
  def train_step(self, state, action, reward, next_state, game_over):
    state = torch.tensor(state, dtype=torch.float)
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