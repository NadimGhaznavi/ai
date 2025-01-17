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
    parser.add_argument('-b1n', '--b1_nodes', type=int, help='Number of nodes in the first block 1 layer.')
    parser.add_argument('-b1l', '--b1_layers', type=int, default=1, help='Number of hidden block 1 layers.')
    parser.add_argument('-b2n', '--b2_nodes', type=int, default=0, help='Number of nodes in the hidden block 2 layer(s).')
    parser.add_argument('-b2l', '--b2_layers', type=int, default=0, help='Number of hidden block 2 layers.')
    parser.add_argument('-b3n', '--b3_nodes', type=int, default=0, help='Number of nodes in the block 3 hidden layer(s).')
    parser.add_argument('-b3l', '--b3_layers', type=int, default=0, help='Number of block 3 hidden layers.')
    parser.add_argument('-e', '--epsilon', type=int, help='Epsilon value for exploration.')
    parser.add_argument('-mg', '--max_games', type=int, default=0, help='Exit the simulation after max_games games.')
    parser.add_argument('-ms', '--max_score', type=int, default=0, help='Exit the simulation if a score of max_score is achieved.')
    parser.add_argument('-msn', '--max_score_num', type=int, default=0, help='Exit the simulation if a score of max_score is achieved max_num times.')
    parser.add_argument('-ns', '--nu_score', type=int, default=0, help='The nu algorithm is triggered when the score exceeds nu_score.')
    parser.add_argument('-nv', '--nu_value', type=int, default=0, help='The initial amount of randomness the nu algorithm injects.')
    parser.add_argument('-s', '--speed', type=int, default=0, help='Set the game speed.')
    parser.add_argument('-sd', '--sim_data_dir', type=str, default=None, help='Set a custom directory to store simulation results.')
    parser.add_argument('-v', '--ai_version', type=int, default=None, help='Load a previous simulation with version ai_version.')

    # Parse arguments
    args = parser.parse_args()

    # Access the INI file 
    self.config = configparser.ConfigParser()
    if not os.path.isfile(ini_file):
      print(f"ERROR: Cannot find INI file ({ini_file}), exiting")
    self.config.read(ini_file)
    
    # Override INI file settings if values were passed in via command line switches
    default = self.config['default']
    if args.b1_nodes:
      default['b1_nodes'] = str(args.b1_nodes)
    if args.b1_layers:
      default['b1_layers'] = str(args.b1_layers)
    if args.b2_nodes:
      default['b2_nodes'] = str(args.b2_nodes)
    if args.b2_layers:
      default['b2_layers'] = str(args.b2_layers)
    if args.b3_nodes:
      default['b3_nodes'] = str(args.b3_nodes)
    if args.b3_layers:
      default['b3_layers'] = str(args.b3_layers)
    if args.sim_data_dir:
      default['sim_data_dir'] = args.sim_data_dir
    if args.epsilon:
      default['epsilon_value'] = str(args.epsilon)
    if args.max_games:
      default['max_games'] = str(args.max_games)
    if args.max_score:
      default['max_score'] = str(args.max_score)
    if args.max_score_num:
      default['max_score_num'] = str(args.max_score_num)
    if args.nu_score:
      default['nu_score'] = str(args.nu_score)
    if args.nu_value:
      default['nu_value'] = str(args.nu_value)
    if args.speed:
      default['game_speed'] = str(args.speed)
    if args.ai_version:
      default['ai_version'] = str(args.ai_version)

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
                      'sim_save_checkpoint_freq', 'status_iter', 'top_margin',
                      'nu_score', 'nu_value']
    # Key/value pairs where the value is a float
    float_values = ['discount', 'learning_rate']
    # Key/value pairs where the value is a boolean
    boolean_values = ['enable_relu']
    # For all other key/value pairs, the value is a string.
    value = self.config['default'][key]

    if key in integer_values:
      return int(value) # Return an int
    elif key in float_values:
      return float(value) # Return a float
    elif key in boolean_values:
      return bool(value) # Return a boolean
    else:
      return value # Return a string
  