"""
LinearQNet.py

This file contains a sub-class of nn.Module that is used to house the 
model that the AI agent uses when playing the Snake Game.
"""
import torch
import torch.nn as nn
import os, sys

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig

class LinearQNet(nn.Module):
  #def __init__(self, in_features, 
  #             b1_nodes, b1_layers, 
  #             b2_nodes, b2_layers,
  #             b3_nodes, b3_layers,
  #             out_features, ai_version):
  def __init__(self, ini, log, model_level):
    """
    The class accepts the following parameters:

        * in_features -> int  : This is the size of the Snake game state array
        * b1_nodes -> int     : The number of nodes used when creating a Linear layer
        * b1_layers -> int    : The number of hidden layers with b1_nodes
        * b2_nodes -> into    : The number of nodes used in the B2 block layers
        * b2_layers -> int    : The number of hidden layers with b2_nodes
        * b3_nodes -> int     : The number of nodes used in the B3 block layers
        * b3_layers -> into   : The number of hidden layers with b3_nodes
        * out_features -> int : The number of nodes in the output layer
        * ai_version -> int   : The version number of the instance configuration

    The class returns a nn.Sequential model that contains the following:

    SnakeModel(
        (layer_stack): Sequential(
            (0): ReLU()
            (1): Linear(in_features=40, out_features=b1_nodes, bias=True)
            ### Start of B1 block ------------
            (2): Sequential(
            (0): ReLU()
            (1): Linear(in_features=b1_nodes, out_features=[b2_nodes|out_features], bias=True)
            )
            ### Start of B2 BLOCK (only present if b2_layers > 0)
            (3): Sequential(
            (0): ReLU()
            (1): Linear(in_features=b2_nodes, out_features=[b3_nodes|out_features], bias=True)
            )
            ### Start of B3 BLOCK (only present if b3_layers > 0)
            (4): Sequential(
            (0): ReLU()
            (1): Linear(in_features=b3_nodes, out_features=3, bias=True)
            ### 
            )
        )
    )

    Note that the pattern of the B2 block (ReLU followed by Linear) is repeated
    b2_layers times. If b2_layers is 0, then this block and the B3 block are not
    present.

    The same is true of the B3 layer i.e. it is only present if b3_layers > 0 and
    the pattern (ReLU followed by Linear) is repeated b3_layers times.
    """  
    super().__init__()
    self.log = log
    # This should be "1" or "2" to indicate whether this is a level 1 or level 2 model
    self.level = model_level
    
    # Set this random seed so things are repeatable. Also set this so that we
    # can change the random seed to make sure the overall results are consistent,
    # even with a different random seed i.e. not due to random chance. I.e. 
    # repeat a simulation with a different random seed to get the same results.
    torch.manual_seed(ini.get('random_seed'))

    # A counter to keep track of the number of steps in a game where this model
    # was used. It will get reset at the beginning of each game.
    self.steps = 0
    self.total_steps = 0 # A total step count that only increases
    self.in_features = ini.get('in_features')
    # Level 1 neural network
    self.b1_nodes = ini.get('b1_nodes')
    self.b1_layers = ini.get('b1_layers')
    self.b2_nodes = ini.get('b2_nodes')
    self.b2_layers = ini.get('b2_layers')
    self.b3_nodes = ini.get('b3_nodes')
    self.b3_layers = ini.get('b3_layers')
    self.out_features = ini.get('out_features')
    self.dropout_p = ini.get('dropout_p')
    self.dropout_min = ini.get('dropout_min')
    self.dropout_max = ini.get('dropout_max')
    self.dropout_static = ini.get('dropout_static')
    self.p_value = 0

    # Enable dropout layers in the model if the user has specified a dynamic
    # ( --dropout_p, --dropout_min and --dropout_max) or static ( --dropout_staic)
    # configuration
    if self.dropout_static > 0:
      self.dropout = True
    else:
      self.dropout = False

    # The basic main model framework
    main_block = nn.Sequential()

    main_block.append(nn.Sequential()) # input block
    main_block.append(nn.Sequential()) # B1 block
    if self.b2_layers > 0:
      main_block.append(nn.Sequential()) # B2 block
    if self.b3_layers > 0:
      main_block.append(nn.Sequential()) # B3 block
    main_block.append(nn.Sequential()) # output block

    # Input layer
    main_block[0].append(nn.Linear(in_features=self.in_features, out_features=self.b1_nodes))
    if self.dropout:
      main_block[0].append(nn.Dropout(p=self.p_value))

    ## B1 Block
    if self.b2_layers > 0:
      # With a B2 block
      layer_count = 0
      while layer_count < self.b1_layers:
        if self.dropout and layer_count > 1:
          main_block[1].append(nn.Dropout(p=self.p_value))
        main_block[1].append(nn.ReLU())
        main_block[1].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b1_nodes))
        layer_count += 1
      main_block[1].append(nn.ReLU())
      main_block[1].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b2_nodes))
    else:
      # With no B2 block
      layer_count = 0
      while layer_count < self.b1_layers:
        main_block[1].append(nn.ReLU())
        main_block[1].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b1_nodes))
        layer_count += 1
        if self.dropout and layer_count < self.b1_layers:
          main_block[1].append(nn.Dropout(p=self.p_value))

    ## B2 Block
    if self.b2_layers > 0:
      if self.b3_layers > 0:
        # With a B3 block
        layer_count = 0
        while layer_count < self.b2_layers:
          main_block[2].append(nn.ReLU())
          main_block[2].append(nn.Linear(in_features=self.b2_nodes, out_features=self.b2_nodes))        
          if self.dropout:
            main_block[2].append(nn.Dropout(p=self.p_value))
          layer_count += 1
        if self.p_value:
          main_block[2].append(nn.Dropout(p=self.p_value))
        main_block[2].append(nn.ReLU())
        main_block[2].append(nn.Linear(in_features=self.b2_nodes, out_features=self.b3_nodes))
      else:
        # with no B3 block
        layer_count = 0
        while layer_count < self.b2_layers and layer_count != self.b2_layers:
          main_block[2].append(nn.ReLU())
          main_block[2].append(nn.Linear(in_features=self.b2_nodes, out_features=self.b2_nodes))
          layer_count += 1
    
    ## B3 Block
    if self.b3_layers > 0:
      layer_count = 0
      while layer_count < self.b3_layers:
        main_block[3].append(nn.ReLU())
        main_block[3].append(nn.Linear(in_features=self.b3_nodes, out_features=self.b3_nodes))
        layer_count += 1
        if self.dropout and layer_count != self.b3_layers:
          main_block[3].append(nn.Dropout(p=self.dropout))
    
    ## Output block
    if self.b2_layers == 0:
      # Only a B1 block
      main_block[1].append(nn.Linear(in_features=self.b1_nodes, out_features=self.out_features))
    elif self.b3_layers == 0:
      # B1 and B2 layers, no B3 layer
      main_block[2].append(nn.Linear(in_features=self.b2_nodes, out_features=self.out_features))
    else:
      # B1, B2 and B3 layers
      main_block[3].append(nn.Linear(in_features=self.b3_nodes, out_features=self.out_features))
      
    self.main_block = main_block
    self.ascii_print()

  def ascii_print(self):
    ###  An ASCII depiction of the neural network
    self.log.log(f"====== Level {self.level // 10} Neural Network Architecture ==========")
    self.log.log("Layers           Input        Output")
    self.log.log("---------------------------------------------")
    log_msg = ''
    for block in self.main_block:
      for layer in block:
        if isinstance(layer, nn.Dropout):
          log_msg = log_msg + "Dropout layer    {:>5} {:>13}\n".format('', '')
        if isinstance(layer, nn.ReLU):
          log_msg = log_msg + "Activation (ReLU) layer\n"
        if isinstance(layer, nn.Linear):
          in_features = layer.in_features
          out_features = layer.out_features
          log_msg = log_msg + "Linear layer     {:>5} {:>13}\n".format(in_features, out_features)
    self.log.log(log_msg)
    
    if self.dropout:
      log_msg = "Dropout layers, p-value is {:>16}".format(self.p_value)
      self.log.log(log_msg)

  def forward(self, x):
    """
    Default nn.Module behaviour. 
    """
    self.steps += 1
    self.total_steps += 1
    return self.main_block(x)
  
  def get_steps(self):
    """
    Returns the number of steps the AI agent has taken.
    """
    return 'L{} model steps# {:>5}'.format(self.level // 10, self.steps)
  
  def get_total_steps(self):
    """
    Returns the total number of steps the AI agent has taken.
    """
    return 'L{} total model steps# {:>9}'.format(self.level // 10, self.total_steps)

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

