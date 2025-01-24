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
  def __init__(self, config, label, ai_version):
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
    # This should be "1" or "2" to indicate whether this is a level 1 or level 2 model
    self.label = label 
    self.ai_version = ai_version
    
    config = AISnakeGameConfig(ai_version)
    torch.manual_seed(config.get('random_seed'))

    self.in_features = config.get('in_features')
    if self.label == 1:
      # Level 1 neural network
      self.b1_nodes = config.get('b1_nodes')
      self.b1_layers = config.get('b1_layers')
      self.b2_nodes = config.get('b2_nodes')
      self.b2_layers = config.get('b2_layers')
      self.b3_nodes = config.get('b3_nodes')
      self.b3_layers = config.get('b3_layers')
      self.out_features = config.get('out_features')
      self.dropout_p = config.get('dropout_p')
      self.dropout_min = config.get('dropout_min')
      self.dropout_max = config.get('dropout_max')
      self.dropout_static = config.get('dropout_static')
      self.p_value = 0
    elif self.label == 2:
      # Level 2 neural network
      self.b1_nodes = config.get('l2_b1_nodes')
      self.b1_layers = config.get('l2_b1_layers')
      self.b2_nodes = config.get('l2_b2_nodes')
      self.b2_layers = config.get('l2_b2_layers')
      self.b3_nodes = config.get('l2_b3_nodes')
      self.b3_layers = config.get('l2_b3_layers')
      self.out_features = config.get('out_features')
      self.dropout_p = config.get('l2_dropout_p')
      self.dropout_min = config.get('l2_dropout_min')
      self.dropout_max = config.get('l2_dropout_max')
      self.dropout_static = config.get('l2_dropout_static')
      self.p_value = 0

    # Enable dropout layers in the model if the user has specified a dynamic
    # ( --dropout_p, --dropout_min and --dropout_max) or static ( --dropout_staic)
    # configuration
    if self.dropout_min > 0 or self.dropout_static > 0:
      self.dropout = True
      if self.dropout_min:
        # Dynamic dropout configuration
        self.p_value = self.dropout_p
      else:
        # Static droptout configuration 
        self.p_value = self.dropout_static

    else:
      self.dropout = False


    self.ascii_print()

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
    main_block[0].append(nn.ReLU())
    main_block[0].append(nn.Linear(in_features=self.in_features, out_features=self.b1_nodes))
    if self.dropout:
      main_block[0].append(nn.Dropout(p=self.p_value))

    ## B1 Block
    if self.b2_layers > 0:
      # With a B2 block
      layer_count = 1
      while layer_count < self.b1_layers:
        main_block[1].append(nn.ReLU())
        main_block[1].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b1_nodes))
        if self.dropout:
          main_block[1].append(nn.Dropout(p=self.p_value))
        layer_count += 1
      main_block[1].append(nn.ReLU())
      main_block[1].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b2_nodes))
    else:
      # With no B2 block
      layer_count = 1
      while layer_count < self.b1_layers:
        main_block[1].append(nn.ReLU())
        main_block[1].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b1_nodes))
        layer_count += 1      

    ## B2 Block
    if self.b2_layers > 0:
      if self.b3_layers > 0:
        # With a B3 block
        layer_count = 1
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
        main_block[2].append(nn.ReLU())
        main_block[2].append(nn.Linear(in_features=self.b3_nodes, out_features=self.out_features))
        layer_count = 1
        while layer_count < self.b2_layers:
          main_block[2].append(nn.ReLU())
          main_block[2].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b1_nodes))
          layer_count += 1
    
    ## B3 Block
    if self.b3_layers > 0:
      layer_count = 1
      while layer_count < self.b3_layers:
        main_block[3].append(nn.ReLU())
        main_block[3].append(nn.Linear(in_features=self.b3_nodes, out_features=self.b3_nodes))
        layer_count += 1
        if self.dropout and layer_count != self.b3_layers:
          main_block[3].append(nn.Dropout(p=self.dropout))
    
    ## Output block
    if self.b2_layers == 0:
      pass
      # Only a B1 block
      main_block[1].append(nn.ReLU())
      main_block[1].append(nn.Linear(in_features=self.b1_nodes, out_features=self.out_features))
    elif self.b3_layers == 0:
      # B1 and B2 layers, no B3 layer
      main_block[2].append(nn.ReLU())
      main_block[2].append(nn.Linear(in_features=self.b2_nodes, out_features=self.out_features))
    else:
      # B1, B2 and B3 layers
      main_block[3].append(nn.ReLU())
      main_block[3].append(nn.Linear(in_features=self.b3_nodes, out_features=self.out_features))
      
    self.main_block = main_block

  def ascii_print(self):
    ###  An ASCII depiction of the neural network
    print(f"====== Level {self.label} Neural Network Architecture ==========")
    print("Blocks       Nodes   Layers  Total  Nodes")
    print("------------------------------------------------------")
    print("Input block  {:>5} {:>8} {:>13}".format(self.in_features, 1, self.in_features))
    print("B1 block     {:>5} {:>8} {:>13}".format(self.b1_nodes, self.b1_layers, self.b1_nodes*self.b1_layers))
    print("B2 block     {:>5} {:>8} {:>13}".format(self.b2_nodes, self.b2_layers, self.b2_nodes*self.b2_layers))
    print("B3 block     {:>5} {:>8} {:>13}".format(self.b3_nodes, self.b3_layers, self.b3_nodes*self.b3_layers))
    print("Output block {:>5} {:>8} {:>13}".format(self.out_features, 1, self.out_features))
    print("------------------------------------------------------")
    print("Totals                   {:>16}".format(self.in_features + (self.b1_nodes*self.b1_layers) + \
                                                   (self.b2_nodes*self.b2_layers) + \
                                                    (self.b3_nodes*self.b3_layers) + self.out_features))

  def forward(self, x):
    """
    Default nn.Module behaviour. 
    """
    return self.main_block(x)
  
  def has_dynamic_dropout(self):
    """
    Returns True if the network has dynamic dropout layers.
    """
    if self.dropout_min:
      return True
    return False
  def insert_layer(self, block_num):
    # Insert the new layer
    print(f"LinearQNet: Inserting new B{block_num} layer")
    print("----- Before -------------------------------------------------")
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

    print("----- After --------------------------------------------------")
    self.ascii_print()
    
  def load_checkpoint(self, optimizer, load_path):
    """
    Loads the model including the weights, epoch from the 
    load_path file.
    """
    checkpoint = torch.load(load_path, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    self.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
  def load_model(self, optimizer, load_path):
    """
    Load the model from the load path. Do not include the weights 
    or epoch. Initialize the epoch to 0.
    """
    checkpoint = torch.load(load_path, weights_only=True)
    state_dict = checkpoint['model_state_dict']
    self.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    state_dict['num_games'] = 0

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
          print(f"LinearQNet: Setting P value for dropout layer(s) to {p_value}")
          block.p = p_value

