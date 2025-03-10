import yaml
import argparse
import os

CONFIG_FILE = 'AISim.yml'

class SimConfig():

    def __init__(self):
        # Read in the application default settings
        self.load(CONFIG_FILE)

        # Get the next simulation number
        with open(self.get('next_num_file'), 'r') as file:
            for line in file:
                sim_num = int(line.strip())
                self.set('sim_num', sim_num)

        with open(self.get('next_num_file'), 'w') as file:
            file.write(str(self.get('sim_num') + 1))

        # Parse any command line args
        parser = argparse.ArgumentParser(description='AI Simulator')
        parser.add_argument('-bs', '--block_size', default=0, type=int, help='Game board square size.')
        parser.add_argument('-ep', '--epsilon', default=0, type=int, help='Epsilon value.')
        parser.add_argument('-in', '--ini_file', default=None, type=str, help='Initial configuration file.')
        parser.add_argument('-ma', '--max_epochs', default=0, type=int, help='Number of simulations to run.')
        parser.add_argument('-mo', '--model', default=None, type=str, help='Model to use [linear|rnn|t], default linear.')
        parser.add_argument('-ne', '--nu_enabled', default=0, type=int, help='Enable the Nu algorithm.')
        parser.add_argument('-nu', '--nu_max_epochs', default=None, type=int, help='Number of games before disabling the Nu algorithm.')
        parser.add_argument('-re', '--restart', default=None, type=int, help='Restart simulation version RESTART.')
        parser.add_argument('-sp', '--speed', default=0, type=int, help='Set the game speed, default is 500.')

        args = parser.parse_args()
        if args.ini_file:
            self.load(args.ini_file)
            self.set('sim_num', sim_num)
            print(f"Loading configuration from {args.ini_file}")
        if args.nu_enabled == 1:
            self.set('nu_enabled', True)
        if args.block_size:
            self.set('block_size', args.block_size)
        if args.epsilon:
            self.set('epsilon_value', args.epsilon)
        if args.max_epochs:
            self.set('max_epochs', args.max_epochs)
        if args.nu_max_epochs is not None:
            self.set('nu_max_epochs', args.nu_max_epochs)
        if args.model:
            self.set('model', args.model)
        if args.restart:
            self.set('restart', args.restart)
        else:
            self.set('restart', 0)
        if args.speed:
            self.set('game_speed', args.speed)

        self.init()
      
    def __del__(self):
        self.save()
    
    def get(self, key):
        if key in self.config['custom']:
            return self.config['custom'][key]
        elif key in self.config['default']:
            return self.config['default'][key]
        else:
            print(f"ERROR: Can't find key ({key})")
            return self.config['default'][key]

    def init(self):
        # Generate simulation specific file
        DATA_DIR = self.get('data_dir')
        #os.makedirs(DATA_DIR, exist_ok=True)
        SIM_NUM = str(self.get('sim_num'))
        SIM_DATA_DIR = os.path.join(DATA_DIR, SIM_NUM)
        self.set('sim_data_dir', SIM_DATA_DIR)
        os.makedirs(SIM_DATA_DIR, exist_ok=True)
        SIM_FILE = SIM_NUM + ".yml"
        SIM_FILE_PATH = os.path.join(SIM_DATA_DIR, SIM_FILE)
        self.config_file = SIM_FILE_PATH
        self.save()

    def load(self, config_file):
        with open(config_file, 'r') as file:
            self.config_file = config_file
            self.config = yaml.safe_load(file)
            self.config['custom'] = {}

    def set(self, key, value):
        self.config['custom'][key] = value

    def save(self):
        with open(self.config_file, 'w') as file_handle:
            yaml.dump(self.config, file_handle)
        
    
    def __str__(self):
        return str(self.config)
    
