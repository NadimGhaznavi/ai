"""
ModelR.py

This file contains a sub-class of nn.Module that is used to house the 
model that the AI agent uses when playing the Snake Game. It's based on
the LinearQNet code.
"""
import torch
import torch.nn as nn
import os, sys

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig

class ModelR(nn.Module):
  def __init__(self, ini, log, model_level):
    super().__init__()
    self.ini = ini
    self.log = log
    self.level = model_level
    self.rnn_count = 0
    
    torch.autograd.set_detect_anomaly(True)
    
    self.hidden_size = 180
    self.in_features = ini.get('in_features')
    self.out_features = ini.get('out_features')
    self.num_layers = 1

    #self.m_rnn = nn.RNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)
    self.m_in = nn.Linear(in_features=400, out_features=400)
    self.m_out = nn.Linear(in_features=400, out_features=3)
    #self.x = None
    #self.fifo = None
    
    self.steps = 0
    self.total_steps = 0
    #self.rnn = nn.RNN(input_size=self.in_features, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
    #self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.out_features)

  def ascii_print(self):
    ###  An ASCII depiction of the neural network
    self.log.log(f"====== Level {self.level} Neural Network Architecture ==========")
    self.log.log("Layers           Input        Output")
    self.log.log("---------------------------------------------")
    log_msg = ''
    for block in self.main_block:
      for layer in block:
        if isinstance(layer, nn.Dropout):
          log_msg = log_msg + "Dropout layer    {:>5} {:>13}\n".format('', '')
        if isinstance(layer, nn.ReLU):
          log_msg = log_msg + "Activation (ReLU) layer\n"
        if isinstance(layer, nn.ReLU):
          log_msg = log_msg + "Activation (Tanh) layer\n"
        if isinstance(layer, nn.Linear):
          in_features = layer.in_features
          out_features = layer.out_features
          log_msg = log_msg + "Linear layer     {:>5} {:>13}\n".format(in_features, out_features)
    self.log.log(log_msg)
    
    if self.dropout:
      log_msg = "Dropout layers, p-value is {:>16}".format(self.p_value)
      self.log.log(log_msg)

  def forward(self, x):
    #print("DEBUG Step: ", self.steps)
    #print("DEBUG x.size(): ", x.size())
    self.steps += 1
    self.total_steps += 1
    #print("DEBUG Step: ", self.steps)
    print("DEBUG x.size(): ", x.size())
    x = self.m_in(x)
    x = self.m_out(x)
    print("DEBUG x: ", x)
    return x

  def forward2(self, x):
    print("DEBUG Step: ", self.steps)
    print("DEBUG x: ", x)
    print("DEBUG x.size(): ", x.size())
    self.steps += 1
    device = 'cpu'
    
    if self.rnn_count == 0:
      self.x = x.unsqueeze(0)
    else:
      self.x = torch.cat((self.x, x.unsqueeze(0)), dim=0)

    h0 = torch.zeros(self.num_layers, self.x.size(0)).to(device) 
    print("DEBUG h0: ", h0)
    print("DEBUG self.x: ", self.x)
    print("DEBUG self.x.size(): ", self.x.size())

    out, _ = self.rnn(self.x, h0)
    out = out[:, -1, :]
    out = self.fc(out)
    return out


  def forward3(self, x):
    """
    Default nn.Module behaviour. 
    """
    print("DEBUG RNN count: ", self.rnn_count)
    #print("DEBUG x: ", x)
    #print("DEBUG x.shape: ", x.shape)
    #print("DEBUG self.x: ", self.x)
    #x = self.m_in(x)
    if self.rnn_count == 0:
      #print("DEBUG: Setting self.x")
      self.fifo = x.unsqueeze(0)
    else:
      #x = x.unsqueeze(0)
      print("DEBUG x: ", x)
      print("DEBUG x.shape: ", x.shape)
      if x.shape == torch.Size([self.in_features]):
        x = x.unsqueeze(0)
      self.fifo = torch.cat((self.fifo, x))

    print("DEBUG self.fifo: ", self.fifo)
    
    #print("DEBUG self.fifo: ", self.fifo)
    #print("DEBUG self.fifo.shape: ", self.fifo.shape)
    #self.fifo = self.fifo[1:]

    if self.rnn_count == 10:
      #self.x = self.x[self.hidden_size:]
      self.fifo = self.fifo[1:]
      self.rnn_count -= 1

    self.rnn_count += 1
    self.steps += 1
    self.total_steps += 1
    self.fifo = self.fifo
    # (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE)
    inputs = self.fifo.view(self.rnn_count, self.in_features, len(self.fifo))
    print("DEBUG inputs: ", inputs)
    x, h_n = self.m_rnn(inputs)
    
    x = self.m_out(x)
    print("DEBUG OUT x[0][0]: ", x[0][0])
    return x[0][0]
    
  def get_steps(self):
    """
    Returns the number of steps the AI agent has taken.
    """
    return 'L{:>2} model steps# {:>5}'.format(self.level, self.steps)
  
  def get_total_steps(self):
    """
    Returns the total number of steps the AI agent has taken.
    """
    return 'L{:>2} total model steps# {:>9}'.format(self.level, self.total_steps)

  def has_dynamic_dropout(self):
    """
    Returns True if the network has dynamic dropout layers.
    """
    if self.dropout_min:
      return True
    return False
  
  def insert_layer(self, block_num):
    # Insert the new layer
    self.log.log(f"LinearQNet: Inserting new B{block_num} layer")
    self.log.log("----- Before -------------------------------------------------")
    self.ascii_print()

    self.main_block[0].append(nn.ReLU())

    if block_num == 1:
      self.b1_layers += 1
      self.main_block[0].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b1_nodes))

    elif block_num == 2:
      self.b2_layers += 1
      self.main_block[0].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b2_nodes))
      # Replace the output block, because the output layer shape needs to match the new B2 layer
      self.main_block[1] = nn.Sequential()
      # Insert a new layer with the right shape
      self.main_block[1].append(nn.ReLU())
      self.main_block[1].append(nn.Linear(in_features=self.b2_nodes, out_features=self.out_features))

    elif block_num == 3:
      self.b3_layers += 1
      self.main_block[0].append(nn.Linear(in_features=self.b2_nodes, out_features=self.b3_nodes))
      # Replace the output block, because the output layer shape needs to match the new B2 layer
      self.main_block[1] = nn.Sequential()
      # Insert a new layer with the right shape
      self.main_block[1].append(nn.ReLU())
      self.main_block[1].append(nn.Linear(in_features=self.b3_nodes, out_features=self.out_features))

    self.log.log("----- After --------------------------------------------------")
    self.ascii_print()
    
  def restore_model(self, optimizer, load_path):
    """
    Loads the model including the weights, epoch from the 
    load_path file.
    """
    checkpoint = torch.load(load_path, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    self.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
  def reset_steps(self):
    """
    Resets the number of steps to 0. Should be called at the beginning of
    each game.
    """
    self.steps = 0
    
  def save_checkpoint(self, optimizer, save_path):
    """
    Saves the model including the weights, epoch and model version.
    """
    torch.save({
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'weights_only': False
    }, save_path)

  def save_model(self, optimizer, save_path):
    """
    Saves only the model i.e. not including the weights.
    Save the epoch value as zero.
    """
    torch.save({
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'weights_only': True,
        'num_games': 0
    }, save_path)

  def set_p_value(self, p_value):
    if p_value == self.p_value:
      # The P value for the dropout layers is already set to this value
      return
    for layer in self.main_block:
      for block in layer:
        if isinstance(block, nn.Dropout):
          self.log.log(f"LinearQNet: Setting P value for dropout layer(s) to {p_value}")
          block.p = p_value

  def set_level(self, level):
    self.level = level

