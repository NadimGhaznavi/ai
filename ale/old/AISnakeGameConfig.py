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
    parser.add_argument('-b2n', '--b2_nodes', type=int, default=0, help='Number of nodes in the hidden block 2 layer(s).')
    parser.add_argument('-b2l', '--b2_layers', type=int, default=0, help='Number of hidden block 2 layers.')
    parser.add_argument('-b3n', '--b3_nodes', type=int, default=0, help='Number of nodes in the block 3 hidden layer(s).')
    parser.add_argument('-b3l', '--b3_layers', type=int, default=0, help='Number of block 3 hidden layers.')
    parser.add_argument('-c', '--custom_data_dir', type=str, default=None, help='Set a custom directory to store simulation results.')
    parser.add_argument('-d', '--discount', type=float, default=0, help='The Linear Q discount factor.')
    parser.add_argument('-ds', '--dropout_static', type=float, default=0, help='Create dropout layers and set the p value to this value.')
    parser.add_argument('-e', '--epsilon', type=int, default=0, help='Epsilon value for exploration.')
    parser.add_argument('-he', '--headless', type=int, default=1, help='Run the game in headless mode.')
    parser.add_argument('-i', '--ini_file', type=str, default=None, help='The path to the configuration file.')
    parser.add_argument('-l', '--learning_rate', type=float, default=0, help='Optimizer learning rate.')
    parser.add_argument('-mg', '--max_games', type=int, default=0, help='Exit the simulation after max_games games.')
    parser.add_argument('-ms', '--max_score', type=int, default=0, help='Exit the simulation if a score of max_score is achieved.')
    parser.add_argument('-nbg', '--nu_bad_games', type=int, default=0, help='The number of games with no new high score.')
    parser.add_argument('-nps', '--nu_print_stats', type=bool, default=0, help="Print NuAlgo status information in the console.")
    parser.add_argument('-nud', '--nu_disable', type=bool, default=0, help="Disable NuAlgo, useful when restarting simulations.")
    parser.add_argument('-nds', '--nu_disable-games', type=int, default=0, help='Disable the nu algorithm when this many games have been played.')
    parser.add_argument('-nus', '--nu_score', type=int, default=0, help='The nu algorithm is triggered when the score exceeds nu_score.')
    parser.add_argument('-nuv', '--nu_value', type=int, default=0, help='The initial amount of randomness the nu algorithm injects.')
    parser.add_argument('-r', '--random_seed', type=int, default=0, help='Random seed used by random and torch.')
    parser.add_argument('-re', '--restart', type=int, default=0, help='Restart a previous simulation, the argument is the simulation AI version number.')
    parser.add_argument('-s', '--speed', type=int, default=0, help='Set the game speed.')
    
    # Parse arguments
    args = parser.parse_args()

    self.config = configparser.ConfigParser()

    # Read in the INI file
    if args.ini_file:
      if not os.path.isfile(args.ini_file):
        print(f"ERROR: Unable to find configuration file ({args.ini_file}), exiting")
        sys.exit(1)
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
    if args.b2_nodes:
      default['b2_nodes'] = str(args.b2_nodes)
    if args.b2_layers:
      default['b2_layers'] = str(args.b2_layers)
    if args.b3_nodes:
      default['b3_nodes'] = str(args.b3_nodes)
    if args.b3_layers:
      default['b3_layers'] = str(args.b3_layers)
    if args.custom_data_dir:
      default['custom_data_dir'] = args.custom_data_dir
    if args.discount:
      default['discount'] = str(args.discount)
    if args.dropout_static:
      default['dropout_static'] = str(args.dropout_static)
    if args.epsilon:
      default['epsilon_value'] = str(args.epsilon)
    if args.headless == 0:
      default['headless'] = str(0)
    if args.learning_rate:
      default['learning_rate'] = str(args.learning_rate)
    if args.max_games:
      default['max_games'] = str(args.max_games)
    if args.max_score:
      default['max_score'] = str(args.max_score)
    if args.nu_bad_games:
      default['nu_bad_games'] = str(args.nu_bad_games)
    if args.nu_disable_games:
      default['nu_disable_games'] = str(args.nu_disable_games)
    if args.nu_disable:
      default['nu_enable'] = 'False'
    if args.nu_score:
      default['nu_score'] = str(args.nu_score)
    if args.nu_print_stats:
      default['nu_print_stats'] = str(args.nu_print_stats)
    if args.nu_value:
      default['nu_value'] = str(args.nu_value)
    if args.random_seed:
      default['random_seed'] = str(args.random_seed)
    if args.restart:
      default['restart'] = str(args.restart)
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
    integer_values = ['ai_version', 
                      'agent_x_min_score',
                      'b1_nodes', 'b1_layers', 'b1_score', 'b2_nodes', 
                      'b2_layers', 'b2_score', 'b3_nodes', 'b3_layers', 
                      'batch_size', 'board_border', 'board_height', 'board_width', 
                      'dropout_max','dropout_min', 'epsilon_value', 'game_speed', 
                      'headless', 'in_features', 
                      'level_score', 'lose_game_reward'
                      'max_iter', 'max_memory', 'max_moves', 'max_games', 'max_score', 
                      'max_score_num', 
                      'nu_pool', 'nu_score', 'nu_bad_games', 'nu_disable_games', 
                      'nu_high_grace', 'nu_max_zero_scores', 'nu_max_moves', 
                      'out_features', 'random_seed', 'restart', 'score_height', 
                      'reward_food', 'reward_excessive_move', 'reward_wall_collision',
                      'reward_snake_collision',
                      'sim_save_checkpoint_freq', 'status_iter', 'top_margin',
                      'new_layer_score']
    # Key/value pairs where the value is a float
    float_values = ['discount', 'dropout_p', 'dropout_static', 'learning_rate',
                    'reward_food', 'reward_excessive_move', 'reward_wall_collision',
                      'reward_snake_collision', 'agent_x_dream_reward']
    # Key/value pairs where the value is a boolean
    boolean_values = ['epsilon_enabled', 'epsilon_print_stats', 'print_stats', 'new_simulation',
                      'nu_enable', 'nu_print_stats', 'nu_verbose', 'agent_x_enable_promo',
                      'sim_checkpoint_enable', 'sim_checkpoint_verbose', 'sim_desc_verbose',
                      'steps_stats', 'steps_stats_all', 'steps_verbose',
                      ]
    # For all other key/value pairs, the value is a string.
    value = self.config['default'][key]

    if key in integer_values:
      return int(value) # Return an int
    elif key in float_values:
      return float(value) # Return a float
    elif key in boolean_values:
      if value == 'False' or value == 'false' or value == '0':
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
  
