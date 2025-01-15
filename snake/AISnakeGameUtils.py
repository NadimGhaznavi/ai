"""
AISnakeGameUtils.py

Some helper functions.
"""
import os, sys
import configparser

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from LinearQNet import Linear_QNet
from AISnakeGameConfig import AISnakeGameConfig

def get_new_model(config):
  return Linear_QNet(
    int(config['default']['in_features']), 
    int(config['default']['b1n']), int(config['default']['b1l']), 
    int(config['default']['b2n']), int(config['default']['b2l']), 
    int(config['default']['b3n']), int(config['default']['b3l']), 
    int(config['default']['out_features']),
    bool(config['default']['enable_relu']),
    int(config['default']['ai_version']))

def get_sim_desc(ai_version):
  ini = AISnakeGameConfig()
  sim_desc_basename = ini.sim_desc_basename()
  sim_model_dir = ini.sim_model_dir()
  sim_desc_file = sim_desc_basename + str(ai_version) + '.txt'
  sim_desc_file = os.path.join(sim_model_dir, sim_desc_file)
  if not os.path.isfile(sim_desc_file):
    print(f"ERROR: Unable to find simulation description file ({sim_desc_file}), exiting")
    sys.exit(1)
  config = configparser.ConfigParser()
  config.read(sim_desc_file)
  return config