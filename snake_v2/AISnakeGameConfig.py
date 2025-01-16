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
    self.config = configparser.ConfigParser()
    if not os.path.isfile(ini_file):
      print(f"ERROR: Cannot find INI file ({ini_file}), exiting")
    self.config.read(ini_file)
    
    # Exit the simulation after max_games games
    self._max_games = args.max_games
    
    # Exit the simulation if a score of max_score is achieved
    self._max_score = args.max_score
    
    # Exit the simulation if a score of max_score is achieved max_score_num times
    self._max_score_num = args.max_score_num

    # Set a custom metrics directory
    self._sim_metrics_dir = args.metrics_dir

    # Override INI file settings if values were passed in via command line switches
    if args.b1_nodes:
      self.config['default']['b1_nodes'] = str(args.b1_nodes)
    if args.epsilon:
      self.config['default']['epsilon_value'] = str(args.epsilon)
    if args.max_games:
      self.config['default']['max_games'] = str(args.max_games)
    if args.max_score:
      self.config['default']['max_score'] = str(args.max_score)
    if args.max_score_num:
      self.config['default']['max_score_num'] = str(args.max_score_num)
    if args.metrics_dir:
      self.config['default']['sim_metrics_dir'] = args.metrics_dir

  def get(self, key):
    """
    Return a value for a given key. The key/value pairs are stored in the
    AISnakeGame.ini file, but some of them can be overriden by passing in
    a command line switch.

    All of the values are stored asstrings, so this function also maintains
    lists of integer, float and boolean values. The latter are returned 
    as an integer, float, or boolean, respectively. If you add new integer, float 
    or boolean value to the INI file, then you must also update the lists in this
    function with the appropriate key names.
    """
    # Key/value pairs where the value is an integer
    integer_values = ['ai_version', 'b1_nodes', 'b1_layers', 'b2_nodes', 'b2_layers', 
                      'b3_nodes', 'b3_layers', 'batch_size', 'board_border', 'board_height', 
                      'board_width', 'epsilon_value', 'game_speed', 'in_features', 
                      'max_iter', 'max_memory', 'max_moves', 'max_games', 'max_score', 
                      'max_score_num', 'out_features', 'random_seed', 'score_height', 
                      'sim_save_checkpoint_freq', 
                      'status_iter', 'top_margin']
    # Key/value pairs where the value is a float
    float_values = ['discount', 'learning_rate']
    # Key/value pairs where the value is a boolean
    boolean_values = ['enable_relu']
    # For all other key/value pairs, the value is a string.
    value = self.config['default'][key]

    if key in integer_values:
      return int(value)
    elif key in float_values:
      return float(value)
    elif key in boolean_values:
      return bool(value)
    else:
      return value
  
  