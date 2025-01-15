"""
AISnakeGameConfig.py

This class provides support for parsing command line arguments and loading
common settings from a configuration file instead of being hard coded as
constants into the code.
"""

# Import supporting modules
import os, sys
import argparse
import configparser

# The directory that this script is in
base_dir = os.path.dirname(__file__)

# Global variables
ini_file = os.path.join(base_dir, 'AISnakeGame.ini')

class AISnakeGameConfig():

  def __init__(self):
    # Setup the expected script arguments
    parser = argparse.ArgumentParser(description='AI Snake Game')
    parser.add_argument('-e', '--epsilon', type=int, help='epsilon value for exploration')
    parser.add_argument('-b1n', '--b1_nodes', type=int, help='number of nodes in the first block 1 layer')
    parser.add_argument('-b1l', '--b1_layers', type=int, default=1, help='number of hidden block 1 layers')
    parser.add_argument('-b2n', '--b2_nodes', type=int, default=0, help='number of nodes in the hidden block 2 layer(s)')
    parser.add_argument('-b2l', '--b2_layers', type=int, default=0, help='number of hidden block 2 layers')
    parser.add_argument('-b3n', '--b3_nodes', type=int, default=0, help='number of nodes in the block 3 hidden layer(s)')
    parser.add_argument('-b3l', '--b3_layers', type=int, default=0, help='number of block 3 hidden layers')
    parser.add_argument('--max_games', type=int, default=0, help='exit the simulation after max_games games')
    parser.add_argument('--max_score', type=int, default=0, help='exit the simulation if a score of max_score is achieved')
    parser.add_argument('--max_score_num', type=int, default=0, help='exit the simulation if a score of max_score is achieved max_num times')
    parser.add_argument('--metrics_dir', type=str, default=None, help='set a custom metrics directory')
    parser.add_argument('-v', '--ai_version', type=int, default=None, help='number of block 3 hidden layers')

    # Parse arguments
    args = parser.parse_args()

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
    if not os.path.isfile(ini_file):
      print(f"ERROR: Cannot find INI file ({ini_file}), exiting")
    config.read(ini_file)
    
    # Exit the simulation after max_games games
    self._max_games = args.max_games
    
    # Exit the simulation if a score of max_score is achieved
    self._max_score = args.max_score
    
    # Exit the simulation if a score of max_score is achieved max_score_num times
    self._max_score_num = args.max_score_num

    # Set a custom metrics directory
    self._sim_metrics_dir = args.metrics_dir

    # Read the INI file settings
    self._ai_version_file = config['default']['ai_version_file']
    self._batch_size = config['default']['batch_size']
    self._board_height = config['default']['board_height']
    self._board_width = config['default']['board_width']
    self._discount = config['default']['discount']
    self._enable_relu = config['default']['enable_relu']
    if not args.epsilon:
      self._epsilon_value = config['default']['epsilon_value']
    self._game_speed = config['default']['game_speed']
    self._in_features = config['default']['in_features']
    self._learning_rate = config['default']['learning_rate']
    self._max_iter = config['default']['max_iter']
    self._max_memory = config['default']['max_memory']
    self._max_moves = config['default']['max_moves']
    if not args.max_games:
      self._max_games = config['default']['max_games']
    if not args.max_score:
      self._max_score = config['default']['max_score']
    if not args.max_score_num:
      self._max_score_num = config['default']['max_score_num']
    self._out_features = config['default']['out_features']
    self._random_seed = config['default']['random_seed']
    self._sim_checkpoint_basename = config['default']['sim_checkpoint_basename']
    self._sim_checkpoint_dir = config['default']['sim_checkpoint_dir']
    self._sim_checkpoint_file_suffix = config['default']['sim_checkpoint_file_suffix']
    self._sim_desc_basename = config['default']['sim_desc_basename']
    self._sim_highscore_basename = config['default']['sim_highscore_basename']
    if not args.metrics_dir:
      self._sim_metrics_dir = config['default']['sim_metrics_dir']
    self._sim_model_basename = config['default']['sim_model_basename']
    self._sim_model_dir = config['default']['sim_model_dir']
    self._sim_model_file_suffix = config['default']['sim_model_file_suffix']
    self._sim_save_checkpoint_freq = config['default']['sim_save_checkpoint_freq']
    self._status_iter = config['default']['status_iter']    
  
  def ai_version(self):
    if self._ai_version:
      return int(self._ai_version)
    else:
      return None
  
  def ai_version_file(self):
    return self._ai_version_file
  
  def b1_nodes(self):
    if self._b1_nodes:
      return int(self._b1_nodes)
    else:
      return 0

  def b1_layers(self):
    return int(self._b1_layers)

  def b2_nodes(self):
    return int(self._b2_nodes)

  def b2_layers(self):
    return int(self._b2_layers)

  def b3_nodes(self):
    return int(self._b3_nodes)

  def b3_layers(self):
    return int(self._b3_layers)
  
  def batch_size(self):
    return int(self._batch_size)

  def board_height(self):
    return int(self._board_height)

  def board_width(self):
    return int(self._board_width)
  
  def discount(self):
    return float(self._discount)
  
  def enable_relu(self):
    return bool(self._enable_relu)

  def epsilon_value(self):
    return int(self._epsilon_value)

  def game_speed(self):
    return int(self._game_speed)
  
  def in_features(self):
    return int(self._in_features)

  def learning_rate(self):
    return float(self._learning_rate)

  def max_games(self):
    return int(self._max_games)
  
  def max_iter(self):
    return int(self._max_iter)
  
  def max_memory(self):
    return int(self._max_memory)

  def max_moves(self):
    return int(self._max_moves)
  
  def max_score(self):
    return int(self._max_score)

  def max_score_num(self):
    return int(self._max_score_num)

  def out_features(self):
    return int(self._out_features)

  def random_seed(self):
    return int(self._random_seed)
  
  def save_checkpoint_freq(self):
    return int(self._sim_save_checkpoint_freq)
  
  def sim_checkpoint_basename(self):
    return self._sim_checkpoint_basename
  
  def sim_checkpoint_dir(self):
    return self._sim_checkpoint_dir

  def sim_checkpoint_file_suffix(self):
    return self._sim_checkpoint_file_suffix
  
  def sim_highscore_basename(self):
    return self._sim_highscore_basename
  
  def sim_metrics_dir(self):
    return self._sim_metrics_dir
  
  def sim_model_basename(self):
    return self._sim_model_basename
  
  def sim_desc_basename(self):
    return self._sim_desc_basename
  
  def sim_model_dir(self):
    return self._sim_model_dir
  
  def sim_model_file_suffix(self):
    return self._sim_model_file_suffix
  
  def sim_save_checkpoint_freq(self):
    return int(self._sim_save_checkpoint_freq)
  
  def status_iter(self):
    return int(self._status_iter)
  