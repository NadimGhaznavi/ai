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

# The default INI file
base_dir = os.path.dirname(__file__)
default_ini_file = os.path.join(base_dir, 'AISnakeGame.ini')

class AISnakeGameConfig():

  def __init__(self):
    # Parse command line options
    parser = argparse.ArgumentParser(description='AI Snake Game')
    parser.add_argument('-b1n', '--b1_nodes', type=int, help='Number of nodes in the first block 1 layer.')
    parser.add_argument('-b1l', '--b1_layers', type=int, default=0, help='Number of hidden block 1 layers.')
    parser.add_argument('-b1s', '--b1_score', type=int, default=0, help='Insert a B1 layer when reaching this score.')
    parser.add_argument('-b2n', '--b2_nodes', type=int, default=0, help='Number of nodes in the hidden block 2 layer(s).')
    parser.add_argument('-b2l', '--b2_layers', type=int, default=0, help='Number of hidden block 2 layers.')
    parser.add_argument('-b2s', '--b2_score', type=int, default=0, help='Insert a B2 layer when reaching this score.')
    parser.add_argument('-b3n', '--b3_nodes', type=int, default=0, help='Number of nodes in the block 3 hidden layer(s).')
    parser.add_argument('-b3l', '--b3_layers', type=int, default=0, help='Number of block 3 hidden layers.')
    parser.add_argument('-b3s', '--b3_score', type=int, default=0, help='Insert a B3 layer when reaching this score.')
    parser.add_argument('-d', '--discount', type=float, default=0, help='The Linear Q discount factor.')
    parser.add_argument('-do', '--dropout_p', type=float, default=0, help='Insert a Droput layer with this p value, used with the --dropout_score switch.')
    parser.add_argument('-dss', '--dropout_static', type=float, default=0, help='Create dropout layers and set the p value to this value.')
    parser.add_argument('-dsx', '--dropout_min', type=int, default=0, help='Activate the p value of the droput layer when reaching this score.')
    parser.add_argument('-dsy', '--dropout_max', type=int, default=0, help='Deactivate the p value of the droput layer when reaching this score.')
    parser.add_argument('-e', '--epsilon', type=int, default=0, help='Epsilon value for exploration.')
    parser.add_argument('-l2e', '--l2_epsilon', type=int, default=0, help='Level 2 epsilon value for exploration.')
    parser.add_argument('-i', '--ini_file', type=str, default='AISnakeGame.ini', help='The path to the configuration file.')
    parser.add_argument('-l', '--learning_rate', type=float, default=0, help='Optimizer learning rate.')
    parser.add_argument('-l2b1n', '--l2_b1_nodes', type=int, default=0, help='Number of nodes in the first level 2, block 1 layer.')
    parser.add_argument('-l2b1l', '--l2_b1_layers', type=int, default=0, help='Number of hidden level 2, block 1 layers.')
    parser.add_argument('-l2b1s', '--l2_b1_score', type=int, default=0, help='Insert a level 2, B1 layer when reaching this score.')
    parser.add_argument('-l2b2n', '--l2_b2_nodes', type=int, default=0, help='Number of nodes in the hidden level 2, block 2 layer(s).')
    parser.add_argument('-l2b2l', '--l2_b2_layers', type=int, default=0, help='Number of hidden level 2, block 2 layers.')
    parser.add_argument('-l2b2s', '--l2_b2_score', type=int, default=0, help='Insert a level 2, B2 layer when reaching this score.')
    parser.add_argument('-l2b3n', '--l2_b3_nodes', type=int, default=0, help='Number of nodes in the level 2, block 3 hidden layer(s).')
    parser.add_argument('-l2b3l', '--l2_b3_layers', type=int, default=0, help='Number of level 2, block 3 hidden layers.')
    parser.add_argument('-l2b3s', '--l2_b3_score', type=int, default=0, help='Insert a level 2, B3 layer when reaching this score.')
    parser.add_argument('-l2dss', '--l2_dropout_static', type=float, default=0, help='Create dropout layers and set the p value to this value.')
    parser.add_argument('-l2s', '--l2_score', type=int, default=0, help='The level 2 AI is triggered when the score exceeds l2_score.')
    parser.add_argument('-mg', '--max_games', type=int, default=0, help='Exit the simulation after max_games games.')
    parser.add_argument('-ms', '--max_score', type=int, default=0, help='Exit the simulation if a score of max_score is achieved.')
    parser.add_argument('-msn', '--max_score_num', type=int, default=0, help='Exit the simulation if a score of max_score is achieved max_num times.')
    parser.add_argument('-nls', '--new_layer_score', type=int, default=0, help='Drop in a new layer at this score')
    parser.add_argument('-nbg', '--nu_bad_games', type=int, default=0, help='The number of games with no new high score.')
    parser.add_argument('-nmm', '--nu_max_moves', type=int, default=0, help="Maximum number of random moves injected by NuAlgo.")
    parser.add_argument('-nps', '--nu_print_stats', type=bool, default=0, help="Print NuAlgo status information in the console.")
    parser.add_argument('-ns', '--nu_score', type=int, default=0, help='The nu algorithm is triggered when the score exceeds nu_score.')
    parser.add_argument('-nv', '--nu_value', type=int, default=0, help='The initial amount of randomness the nu algorithm injects.')
    parser.add_argument('-nvm', '--nu_value_max', type=int, default=0, help='Number of random moves to add to the nu pool if nu_num_games_same_score_count_max is exceeded')
    parser.add_argument('-r', '--random_seed', type=int, default=0, help='Random seed used by random and torch.')
    parser.add_argument('-rl1', '--restore_l1', type=str, default=None, help='Load a previous L1 model.')
    parser.add_argument('-rl2', '--restore_l2', type=str, default=None, help='Load a previous L2 model.')
    parser.add_argument('-s', '--speed', type=int, default=0, help='Set the game speed.')
    parser.add_argument('-sd', '--custom_data_dir', type=str, default=None, help='Set a custom directory to store simulation results.')
    
    # Parse arguments
    args = parser.parse_args()

    self.config = configparser.ConfigParser()

    # Read in the INI file
    if args.ini_file:
      self.config.read(args.ini_file)
      self.set_value('ini_file', args.ini_file)
    else:
      self.config.read(default_ini_file)
      self.set_value('ini_file', default_ini_file)

    # Get a new ai_version number
    ai_version = self.get_next_ai_version_num()
    self.set_value('ai_version', str(ai_version))

    # Override INI file settings if values were passed in via command line switches
    default = self.config['default']
    if args.b1_nodes:
      default['b1_nodes'] = str(args.b1_nodes)
    if args.b1_layers:
      default['b1_layers'] = str(args.b1_layers)    
    if args.b1_score:
      default['b1_score'] = str(args.b1_score)    
    if args.b2_nodes:
      default['b2_nodes'] = str(args.b2_nodes)
    if args.b2_layers:
      default['b2_layers'] = str(args.b2_layers)
    if args.b2_score:
      default['b2_score'] = str(args.b2_score)    
    if args.b3_nodes:
      default['b3_nodes'] = str(args.b3_nodes)
    if args.b3_layers:
      default['b3_layers'] = str(args.b3_layers)
    if args.b3_score:
      default['b3_score'] = str(args.b3_score)
    if args.custom_data_dir:
      default['custom_data_dir'] = args.custom_data_dir
    if args.discount:
      default['discount'] = str(args.discount)
    if args.dropout_p:
      default['dropout_p'] = str(args.dropout_p)
    if args.dropout_max:
      default['dropout_max'] = str(args.dropout_max)
    if args.dropout_min:
      default['dropout_min'] = str(args.dropout_min)
    if args.dropout_static:
      default['dropout_static'] = str(args.dropout_static)
    if args.epsilon:
      default['epsilon_value'] = str(args.epsilon)
    if args.l2_b1_nodes:
      default['l2_b1_nodes'] = str(args.l2_b1_nodes)
    if args.l2_b1_layers:
      default['l2_b1_layers'] = str(args.l2_b1_layers)
    if args.l2_b1_score:
      default['l2_b1_score'] = str(args.l2_b1_score)
    if args.l2_b2_nodes:
      default['l2_b2_nodes'] = str(args.l2_b2_nodes)
    if args.l2_b2_layers:
      default['l2_b2_layers'] = str(args.l2_b2_layers)
    if args.l2_b2_score:
      default['l2_b2_score'] = str(args.l2_b2_score)
    if args.l2_b3_nodes:
      default['l2_b3_nodes'] = str(args.l2_b3_nodes)
    if args.l2_b3_layers:
      default['l2_b3_layers'] = str(args.l2_b3_layers)
    if args.l2_b3_score:
      default['l2_b3_score'] = str(args.l2_b3_score)
    if args.l2_dropout_static:
      default['l2_dropout_static'] = str(args.l2_dropout_static)
    if args.l2_epsilon:
      default['l2_epsilon_value'] = str(args.l2_epsilon)
    if args.l2_score:
      default['l2_score'] = str(args.l2_score)
    if args.learning_rate:
      default['learning_rate'] = str(args.learning_rate)
    if args.max_games:
      default['max_games'] = str(args.max_games)
    if args.max_score:
      default['max_score'] = str(args.max_score)
    if args.max_score_num:
      default['max_score_num'] = str(args.max_score_num)
    if args.new_layer_score:
      default['new_layer_score'] = str(args.new_layer_score)
    if args.nu_bad_games:
      default['nu_bad_games'] = str(args.nu_bad_games)
    if args.nu_max_moves:
      default['nu_max_moves'] = str(args.nu_max_moves)
    if args.nu_score:
      default['nu_score'] = str(args.nu_score)
    if args.nu_print_stats:
      default['nu_print_stats'] = str(args.nu_print_stats)
    if args.nu_value:
      default['nu_value'] = str(args.nu_value)
    if args.nu_value_max:
      default['nu_value_max'] = str(args.nu_value_max)
    if args.random_seed:
      default['random_seed'] = str(args.random_seed)
    if args.restore_l1:
      default['restore_l1'] = args.restore_l1
    if args.restore_l2:
      default['restore_l2'] = args.restore_l2
    if args.custom_data_dir:
      default['custom_data_dir'] = args.custom_data_dir
    if args.speed:
      default['game_speed'] = str(args.speed)

    if self.get('b1_nodes') == 0:
      print(f"ERROR: You must set b1_nodes to a non-zero value")
      sys.exit(1)

    if self.get('b1_layers') == 0:
      print(f"ERROR: You must set b1_layers to a non-zero value")
      sys.exit(1)

    if self.get('b2_layers') > 0 and self.get('b2_nodes') == 0:
      print(f"ERROR: You must set b2_nodes to a non-zero value if you set b2_layers")
      sys.exit(1)

    if self.get('b3_layers') > 0 and self.get('b3_nodes') == 0:
      print(f"ERROR: You must set b3_nodes to a non-zero value if you set b3_layers")
      sys.exit(1)

    # Create the base simulation data directory if it does not exist
    os.makedirs(self.get('sim_data_dir'), exist_ok=True)

    # Create the custom simulation data directory if it was specified
    custom_data_dir = self.get('custom_data_dir')
    if custom_data_dir:
      # Set the simulation data directory to the custom one
      custom_data_dir = os.path.join(self.get('sim_data_dir'), custom_data_dir)
      self.set_value('sim_data_dir', custom_data_dir)
      os.makedirs(custom_data_dir, exist_ok=True)


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
    integer_values = ['ai_version', 'b1_nodes', 'b1_layers', 'b1_score', 'b2_nodes', 
                      'b2_layers', 'b2_score', 'b3_nodes', 'b3_layers', 'b3_score', 
                      'batch_size', 'board_border', 'board_height', 'board_width', 
                      'dropout_max','dropout_min', 'epsilon_value', 'game_speed', 
                      'in_features', 
                      'l2_b1_nodes', 'l2_b1_layers', 'l2_b1_score', 
                      'l2_b2_nodes', 'l2_b2_layers', 'l2_b2_score',
                      'l2_b3_nodes', 'l2_b3_layers', 'l2_b3_score',
                      'l2_dropout_p', 'l2_dropout_max', 'l2_dropout_min', 
                      'l2_epsilon_value', 'l2_score',
                      'max_iter', 'max_memory', 'max_moves', 'max_games', 'max_score', 
                      'max_score_num', 'out_features', 'random_seed', 'score_height', 
                      'sim_save_checkpoint_freq', 'status_iter', 'top_margin',
                      'new_layer_score', 'nu_bad_games', 'nu_max_zero_scores', 
                      'nu_max_moves', 'nu_score', 'nu_value']
    # Key/value pairs where the value is a float
    float_values = ['discount', 'dropout_p', 'dropout_static', 'l2_dropout_static', 'learning_rate']
    # Key/value pairs where the value is a boolean
    boolean_values = ['epsilon_print_stats', 'print_stats', 'new_simulation',
                      'sim_checkpoint_enable', 'sim_checkpoint_verbose', 'sim_desc_verbose',
                      'steps_stats', 'steps_verbose',
                      ]
    # For all other key/value pairs, the value is a string.
    try:
      value = self.config['default'][key]
    except:
      print(f"ERROR: Cannot find setting ({key}) in {self.get('ini_file')}, exiting")
      sys.exit(1)

    if key in integer_values:
      return int(value) # Return an int
    elif key in float_values:
      return float(value) # Return a float
    elif key in boolean_values:
      if value == 'False':
        return False
      else:
        return True
    else:
      return value # Return a string
  
  def get_next_ai_version_num(self):
    """
    Get the next available version number from the ai_version file.
    If the file doesn't exist, write '2' to the file an return '1'.
    """    
    ai_version_file = os.path.join(base_dir, self.get('ai_version_file'))
    if os.path.isfile(ai_version_file):
      file_handle = open(ai_version_file, 'r')
      for line in file_handle:
        ai_version = int(line.strip())
      file_handle.close()
      with open(ai_version_file, 'w') as file_handle:
        file_handle.write(str(ai_version + 1))
        file_handle.close()
    else:
      ai_version = 1
      with open(ai_version_file, 'w') as file_handle:
        file_handle.write('2')
        file_handle.close()
    return ai_version
  
  def load_config(self, ai_version):
    sim_desc_basename = self.get('sim_desc_basename')
    sim_data_dir = self.get('sim_data_dir')
    sim_desc_file = str(ai_version) + sim_desc_basename
    sim_desc_file = os.path.join(sim_data_dir, sim_desc_file)
    if not os.path.isfile(sim_desc_file):
      print(f"ERROR: Unable to find simulation description file ({sim_desc_file}), exiting")
      sys.exit(1)
    self.config.read(sim_desc_file)
    return self
  
  def save_sim_desc(self):
    # Get the filename components
    ai_version = self.get('ai_version')
    data_dir = self.get('sim_data_dir')
    desc_file_basename = self.get('sim_desc_basename')
    # Construct the filename
    desc_file = os.path.join(data_dir, str(ai_version) + desc_file_basename)
    with open(desc_file, 'w') as file_handle:
      self.config.write(file_handle)
    if self.get('sim_desc_verbose'):
      print(f"Saved simulation description ({desc_file})")

  def set_value(self, key, value):
    """
    Set method.
    """
    self.config['default'][key] = value

  def write(self, ini_file):
    """
    Create an INI file with the ConfigParser values.
    """
    print(self.config)
    print(type(self.config))
    print(ini_file)
    self.config.write(ini_file)
  
