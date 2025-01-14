"""
LinearQNet.py

This file contains a sub-class of nn.Module that is used to house the 
model that the AI agent uses when playing the Snake Game.
"""
import torch
import torch.nn as nn

class Linear_QNet(nn.Module):
  def __init__(self, input_nodes, 
               b1_nodes, b1_layers, 
               b2_nodes, b2_layers,
               b3_nodes, b3_layers,
               output_nodes, enable_relu, ai_version):
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
    print("{:>4} * {:>2} = {:>5}".format(b1_nodes, b1_layers, b1_nodes*b1_layers))
    print("{:>4} * {:>2} = {:>5}".format(b2_nodes, b2_layers, b2_nodes*b2_layers))
    print("{:>4} * {:>2} = {:>5}".format(b3_nodes, b3_layers, b3_nodes*b3_layers))
    print(" Nodes    = {:>5}".format((b1_nodes*b1_layers)+(b2_nodes*b2_layers)+(b3_nodes*b3_layers)))
    self.ai_version = ai_version

    # The main model
    main_block = nn.Sequential()

    # Input layer
    if enable_relu:
      main_block.append(nn.ReLU())
    main_block.append(nn.Linear(in_features=input_nodes, out_features=b1_nodes))


    ### B1 Block ------------------------------------------------------
    b1_block = nn.Sequential()
    b1_layer_count = 1
    while b1_layer_count != b1_layers:
      b1_layer_count += 1
      if b1_layer_count != b1_layers:
        # There are more B1 layers...
        if enable_relu:
          b1_block.append(nn.ReLU())
        b1_block.append(nn.Linear(in_features=b1_nodes, out_features=b1_nodes))
    
    # There are no more B1 to B1 layers
    
    # Check if there are any B2 layers
    if b2_layers != 0:
      # There are some B2 layers
      if enable_relu:
        b1_block.append(nn.ReLU())
      b1_block.append(nn.Linear(in_features=b1_nodes, out_features=b2_nodes))
      main_block.append(b1_block)
    else:
      # There are no B2 layers, so append an output layer. Model is complete.
      if enable_relu:
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
    """
    Default nn.Module behaviour. 
    """
    return self.layer_stack(x)
  
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

  def save_checkpoint(self, optimizer, save_path, epoch):
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
    self.state_dict['num_games'] = 0
    torch.save({
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'weights_only': True
    }, save_path)

