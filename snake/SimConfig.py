import yaml
import argparse
import os

CONFIG_FILE = 'AISim.yml'

class SimConfig():

    def __init__(self):
        # Read in the application default settings
        with open(CONFIG_FILE, 'r') as file:
            self.config = yaml.safe_load(file)
            self.config['state'] = {}

        # Get the next simulation number
        with open(self.get('next_num_file'), 'r') as file:
            for line in file:
                self.set('sim_num', int(line.strip()))
        with open(self.get('next_num_file'), 'w') as file:
            file.write(str(self.get('sim_num') + 1))

        # Parse any command line args
        parser = argparse.ArgumentParser(description='AI Simulator')
        parser.add_argument('-m', '--max_epochs', default=0, type=int, help='Number of simulations to run.')
        args = parser.parse_args()
        if args.max_epochs:
            self.set('max_epochs', args.max_epochs)

      
    def __del__(self):
        self.save()
    
    def get(self, key):
        if key in self.config['state']:
            return self.config['state'][key]
        elif key in self.config['default']:
            return self.config['default'][key]
        else:
            print(f"ERROR: Can't find key ({key})")
            return self.config['default'][key]
    
    def incr(self, key):
        if key not in self.config['state']:
            self.config['state'][key] = 0
        self.config['state'][key] += 1

    def set(self, key, value):
        self.config['state'][key] = value

    def save(self):
        # Generate simulation specific file
        DATA_DIR = self.get('data_dir')
        os.makedirs(DATA_DIR, exist_ok=True)
        SIM_DIR = str(self.get('sim_num'))
        SIM_DATA_DIR = os.path.join(DATA_DIR, SIM_DIR)
        self.set('data_dir', SIM_DATA_DIR)
        os.makedirs(SIM_DATA_DIR, exist_ok=True)
        SIM_FILE = SIM_DIR + ".yml"
        SIM_FILE_PATH = os.path.join(SIM_DATA_DIR, SIM_FILE)
        with open(SIM_FILE_PATH, 'w') as file:
            yaml.dump(self.config, file)
    
    def __str__(self):
        return str(self.config)
    
