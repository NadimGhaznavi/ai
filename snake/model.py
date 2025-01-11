import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

MODEL_DIR = './models'
MODEL_FILE = 'ai_model_v'
MODEL_FILE_SUFFIX = 'pth'

class Linear_QNet(nn.Module):
  def __init__(self, input_nodes, 
               b1_nodes, b1_layers, 
               b2_nodes, b2_layers,
               b3_nodes, b3_layers,
               output_nodes, ai_version):
    super().__init__()
    print("{:>4} * {:>2} = {:>5}".format(b1_nodes, b1_layers, b1_nodes*b1_layers))
    print("{:>4} * {:>2} = {:>5}".format(b2_nodes, b2_layers, b2_nodes*b2_layers))
    print("{:>4} * {:>2} = {:>5}".format(b3_nodes, b3_layers, b3_nodes*b3_layers))
    print(" Nodes    = {:>5}".format((b1_nodes*b1_layers)+(b2_nodes*b2_layers)+(b3_nodes*b3_layers)))
    self.ai_version = ai_version
    self.layer_stack = nn.Sequential()
    self.layer_stack.append(nn.Linear(in_features=input_nodes, out_features=b1_nodes))
    self.layer_stack.append(nn.ReLU())
    b1_layer_count = 0
    while b1_layer_count != b1_layers:
      b1_layer_count += 1
      if b1_layer_count == b1_layers:
        # We need to figure out the out_features before we define the final B1layer
        if b2_layers == 0:
          # There are no B2 layers, define the final B1 layer
          self.layer_stack.append(nn.Linear(in_features=b1_nodes, out_features=output_nodes))
        else:
          # There are some B2 layers
          # Define the last B1 layer
          self.layer_stack.append(nn.Linear(in_features=b1_nodes, out_features=b2_nodes))
          self.layer_stack.append(nn.ReLU())
          b2_layer_count = 0
          while b2_layer_count != b2_layers:
            b2_layer_count += 1
            if b2_layer_count == b2_layers:
              # We need to figure out the out_features before we define the final B2 layer
              if b3_layers == 0:
                # There are no B3 layers, define the final B2 layer
                self.layer_stack.append(nn.Linear(in_features=b2_nodes, out_features=output_nodes))
              else:
                # There are some B3 layers
                # Define the last B2 layer
                self.layer_stack.append(nn.Linear(in_features=b2_nodes, out_features=b3_nodes))
                self.layer_stack.append(nn.ReLU())
                b3_layer_count = 0
                while b3_layer_count != b3_layers:
                  b3_layer_count += 1
                  if b3_layer_count == b3_layers:
                    # Final B3 layer
                    self.layer_stack.append(nn.Linear(in_features=b3_nodes, out_features=output_nodes))
                  else:
                    self.layer_stack.append(nn.Linear(in_features=b3_nodes, out_features=b3_nodes))
                    self.layer_stack.append(nn.ReLU())
            else:
              self.layer_stack.append(nn.Linear(in_features=b2_nodes, out_features=b2_nodes))
              self.layer_stack.append(nn.ReLU())
      else:
        self.layer_stack.append(nn.Linear(in_features=b1_nodes, out_features=b1_nodes))
        self.layer_stack.append(nn.ReLU())
  
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
      Q_new = reward[idx]
      if not game_over[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
      target[idx][torch.argmax(action).item()] = Q_new
      
    self.optimizer.zero_grad() # Reset the gradients to zero
    loss = self.criterion(target, pred) # Calculate the loss
    loss.backward() # Backpropagate the loss
    self.optimizer.step() # Adjust the weights