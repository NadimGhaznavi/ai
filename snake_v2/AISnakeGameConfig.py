"""
AISnakeGameConfig.py

This class provides support for parsing command line arguments and loading
common settings from a configuration file instead of being hard coded as
constants into the code.
"""

# Import supporting modules
import os
import argparse
import configparser

# The directory that this script is in
base_dir = os.path.dirname(__file__)

# Global variables
ini_file = base_dir + 'AISnakeGame.ini'
default_environment = 'prod'

class AISnakeGameConfig():

  def __init__(self):
    # Setup the expected script arguments
    parser = argparse.ArgumentParser(description='AI Snake Game')
    parser.add_argument('-d', '--debug', type=int, default=0, help='debug level')
    parser.add_argument('--environ', type=str, default=default_environment, help='prod or dev')
    parser.add_argument('--epsilon', type=int, default=0, help='epsilon value for exploration')
    parser.add_argument('-b1n', '--b1_nodes', type=int, help='number of nodes in the first block 1 layer')
    parser.add_argument('-b1l', '--b1_layers', type=int, default=1, help='number of hidden block 1 layers')
    parser.add_argument('-b2n', '--b2_nodes', type=int, default=0, help='number of nodes in the hidden block 2 layer(s)')
    parser.add_argument('-b2l', '--b2_layers', type=int, default=0, help='number of hidden block 2 layers')
    parser.add_argument('-b3n', '--b3_nodes', type=int, default=0, help='number of nodes in the block 3 hidden layer(s)')
    parser.add_argument('-b3l', '--b3_layers', type=int, default=0, help='number of block 3 hidden layers')
    parser.add_argument('-v', '--ai_version', type=int, default=None, help='number of block 3 hidden layers')

    # Parse arguments
    args = parser.parse_args()
    self._args = args

    # Get the debug level from the command line switch or it's default
    self._debug = args.debug

    # Override the debug level if the AISNAKEGAME_DEBUG variable
    # has been set
    try:
      environ_var = os.environ['AISNAKEGAME_DEBUG']
      self._debug = environ_var
    except KeyError:
      os.environ['AISNAKEGAME_DEBUG'] = self._debug
    
    # prod or dev
    environ = args.environ
    self._environ = environ

    # Block 1, 2, 3 nodes and layers
    self._b1_nodes = args.b1_nodes
    self._b1_layers = args.b1_layers
    self._b2_nodes = args.b2_nodes
    self._b2_layers = args.b2_layers
    self._b3_nodes = args.b3_nodes
    self._b3_layers = args.b3_layers

    # Epsilon value for exploration
    self._epsilon_value = args.epsilon

    # AI version
    self._ai_version = args.ai_version

    # Access the INI file 
    config = configparser.ConfigParser()
    config.read(args.ini_file)

    # Read the INI file settings
    self._ai_version_file = config[environ]['ai_version_file']
    self._batch_size = config[environ]['batch_size']
    self._board_height = config[environ]['board_height']
    self._board_width = config[environ]['board_width']
    self._discount = config[environ]['discount']
    self._enable_relu = config[environ]['enable_relu']
    self._epsilon_value = config[environ]['epsilon_value']
    self._game_speed = config[environ]['game_speed']
    self._in_features = config[environ]['in_features']
    self._learning_rate = config[environ]['learning_rate']
    self._max_iter = config[environ]['max_iter']
    self._max_memory = config[environ]['max_memory']
    self._max_moves = config[environ]['max_moves']
    self._out_features = config[environ]['out_features']
    self._random_seed = config[environ]['random_seed']
    self._sim_checkpoint_basename = config[environ]['sim_checkpoint_basename']
    self._sim_checkpoint_dir = config[environ]['sim_checkpoint_dir']
    self._sim_checkpoint_file_suffix = config[environ]['sim_checkpoint_file_suffix']
    self._sim_model_basename = config[environ]['sim_model_basename']
    self._sim_model_dir = config[environ]['sim_model_dir']
    self._sim_model_file_suffix = config[environ]['sim_model_file_suffix']
    self._sim_save_checkpoint_freq = config[environ]['sim_save_checkpoint_freq']
    self._status_iter = config[environ]['status_iter']
  
  def ai_version(self):
    return self._ai_version
  
  def ai_version_file(self):
    return self._ai_version_file
  
  def b1_nodes(self):
    return self._b1_nodes

  def b1_layers(self):
    return self._b1_layers

  def b2_nodes(self):
    return self._b2_nodes

  def b2_layers(self):
    return self._b2_layers

  def b3_nodes(self):
    return self._b3_nodes

  def b3_layers(self):
    return self._b3_layers

  def board_height(self):
    return self._board_height

  def board_width(self):
    return self._board_width
  
  def discount(self):
    return self._discount
  
  def enable_relu(self):
    return self._enable_relu

  def epsilon_value(self):
    return self._epsilon_value

  def game_speed(self):
    return self._game_speed
  
  def in_features(self):
    return self._in_features

  def learning_rate(self):
    return self._learning_rate

  def max_iter(self):
    return self._max_iter
  
  def max_memory(self):
    return self._max_memory

  def max_moves(self):
    return self._max_moves
  
  def out_features(self):
    return self._out_features

  def random_seed(self):
    return self._random_seed
  
  def sim_checkpoint_basename(self):
    return self._sim_checkpoint_basename
  
  def sim_checkpoint_dir(self):
    return self._sim_checkpoint_dir

  def sim_checkpoint_file_suffix(self):
    return self._sim_checkpoint_file_suffix
  
  def sim_model_basename(self):
    return self._sim_model_basename
  
  def sim_model_dir(self):
    return self._sim_model_dir
  
  def sim_model_file_suffix(self):
    return self._sim_model_file_suffix
  
  def sim_save_checkpoint_freq(self):
    return self._sim_save_checkpoint_freq
  
  def status_iter(self):
    return self._status_iter
  