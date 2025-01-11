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

    # The main model
    main_block = nn.Sequential()

    # Input layer
    main_block.append(nn.Linear(in_features=input_nodes, out_features=b1_nodes))

    ### B1 Block ------------------------------------------------------
    b1_block = nn.Sequential()
    b1_layer_count = 0
    while b1_layer_count != b1_layers:
      b1_layer_count += 1
      if b1_layer_count != b1_layers:
        # There are more B1 layers...
        b1_block.append(nn.ReLU())
        b1_block.append(nn.Linear(in_features=b1_nodes, out_features=b1_nodes))
    
    # There are no more B1 to B1 layers
    
    # Check if there are any B2 layers
    if b2_layers != 0:
      # There are some B2 layers
      b1_block.append(nn.ReLU())
      b1_block.append(nn.Linear(in_features=b1_nodes, out_features=b2_nodes))
      main_block.append(b1_block)
    else:
      # There are no B2 layers, so append an output layer. Model is complete.
      b1_block.append(nn.ReLU())
      b1_block.append(nn.Linear(in_features=b1_nodes, out_features=output_nodes))
      main_block.append(b1_block)

    ### B2 Block -------------------------------------------------------
    b2_block = nn.Sequential()
    b2_layer_count = 0
    while b2_layer_count != b2_layers:
      b2_layer_count += 1
      if b2_layer_count != b2_layers:
        # There are more B2 layers...
        b2_block.append(nn.ReLU())
        b2_block.append(nn.Linear(in_features=b2_nodes, out_features=b2_nodes))
    
    # There are no more B2 to B2 layers

    if b2_layers != 0 and b3_layers != 0:
      # There were some B2 layers and there are some B3 layers
      b2_block.append(nn.ReLU())
      b2_block.append(nn.Linear(in_features=b2_nodes, out_features=b3_nodes))
      main_block.append(b2_block)
    elif b2_layers != 0 and b3_layers == 0:
      # There were some B2 layers and there are no B3 layers. Append an output
      # layer and model is complete.
      b2_block.append(nn.ReLU())
      b2_block.append(nn.Linear(in_features=b2_nodes, out_features=output_nodes))
      main_block.append(b2_block)

    ### B3 Block -------------------------------------------------------
    b3_block = nn.Sequential()
    b3_layer_count = 0
    while b3_layer_count != b3_layers:
      b3_layer_count += 1
      if b3_layer_count != b3_layers:
        # There are more B3 layers...
        b3_block.append(nn.ReLU())
        b3_block.append(nn.Linear(in_features=b3_nodes, out_features=b3_nodes))
      else:
        b3_block.append(nn.ReLU())
        b3_block.append(nn.Linear(in_features=b3_nodes, out_features=output_nodes))
        main_block.append(b3_block)
      
    self.layer_stack = main_block
      
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

  def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

  def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch  
    
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