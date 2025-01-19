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

ini = AISnakeGameConfig()

torch.manual_seed(ini.get('random_seed'))

class Linear_QNet(nn.Module):
  def __init__(self, in_features, 
               b1_nodes, b1_layers, 
               b2_nodes, b2_layers,
               b3_nodes, b3_layers,
               out_features, ai_version):
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

    self.ai_version = ai_version
    self.in_features = in_features
    self.b1_nodes = b1_nodes
    self.b1_layers = b1_layers
    self.b2_nodes = b2_nodes
    self.b2_layers = b2_layers
    self.b3_nodes = b3_nodes
    self.b3_layers = b3_layers
    self.out_features = out_features

    self.ascii_print()

    # The basic main model framework
    main_block = nn.Sequential()

    main_block.append(nn.Sequential()) # input block
    main_block.append(nn.Sequential()) # B1 block
    if b2_layers > 0:
      main_block.append(nn.Sequential()) # B2 block
    if b3_layers > 0:
      main_block.append(nn.Sequential()) # B3 block
    main_block.append(nn.Sequential()) # output block

    # Input layer
    main_block[0].append(nn.ReLU())
    main_block[0].append(nn.Linear(in_features=in_features, out_features=b1_nodes))

    ## B1 Block
    if b2_layers > 0:
      # With a B2 block
      layer_count = 1
      while layer_count < b1_layers:
        main_block[1].append(nn.ReLU())
        main_block[1].append(nn.Linear(in_features=b1_nodes, out_features=b1_nodes))
        layer_count += 1
      main_block[1].append(nn.ReLU())
      main_block[1].append(nn.Linear(in_features=b1_nodes, out_features=b2_nodes))
    else:
      # With no B2 block
      layer_count = 1
      while layer_count < b1_layers:
        main_block[1].append(nn.ReLU())
        main_block[1].append(nn.Linear(in_features=b1_nodes, out_features=b1_nodes))
        layer_count += 1      

    ## B2 Block
    if b2_layers > 0:
      if b3_layers > 0:
        # With a B3 block
        layer_count = 1
        while layer_count < b2_layers:
          layer_count += 1
          main_block[2].append(nn.ReLU())
          main_block[2].append(nn.Linear(in_features=b2_nodes, out_features=b2_nodes))        
        main_block[2].append(nn.ReLU())
        main_block[2].append(nn.Linear(in_features=b2_nodes, out_features=b3_nodes))
      else:
        # with no B3 block
        main_block[2].append(nn.ReLU())
        main_block[2].append(nn.Linear(in_features=b3_nodes, out_features=out_features))
        layer_count = 1
        while layer_count < b2_layers:
          main_block[2].append(nn.ReLU())
          main_block[2].append(nn.Linear(in_features=b1_nodes, out_features=b1_nodes))
          layer_count += 1
    
    ## B3 Block
    if b3_layers > 0:
      layer_count = 1
      while layer_count < b2_layers:
        main_block[3].append(nn.ReLU())
        main_block[3].append(nn.Linear(in_features=b3_nodes, out_features=b3_nodes))
        layer_count += 1
    
    ## Output block
    if b2_layers == 0:
      pass
      # Only a B1 block
      main_block[1].append(nn.ReLU())
      main_block[1].append(nn.Linear(in_features=b1_nodes, out_features=out_features))
    elif b3_layers == 0:
      # B1 and B2 layers, no B3 layer
      main_block[2].append(nn.ReLU())
      main_block[2].append(nn.Linear(in_features=b2_nodes, out_features=out_features))
    else:
      # B1, B2 and B3 layers
      main_block[3].append(nn.ReLU())
      main_block[3].append(nn.Linear(in_features=b3_nodes, out_features=out_features))
      
    self.main_block = main_block

  def ascii_print(self):
    ###  An ASCII depiction of the neural network
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
  
  def insert_layer(self):
    print("LinearQNet: Inserting new layer")
    print("----- Before -------------------------------------------------")
    self.ascii_print()
    self.main_block[0].append(nn.ReLU())
    self.main_block[0].append(nn.Linear(in_features=self.b1_nodes, out_features=self.b1_nodes))
    self.b1_layers += 1
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

  def save_checkpoint(self, optimizer, save_path, num_games):
    """
    Saves the model including the weights, epoch and model version.
    """
    torch.save({
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'weights_only': False,
        'num_games': num_games
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

