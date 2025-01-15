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
  """
  Return a Linear_QNet instance with parameters taken form a 
  configparser.ConfigParser object (which is stored in a 
  simulation description file for each simulation).
  """
  return Linear_QNet(
    int(config['default']['in_features']), 
    int(config['default']['b1n']), int(config['default']['b1l']), 
    int(config['default']['b2n']), int(config['default']['b2l']), 
    int(config['default']['b3n']), int(config['default']['b3l']), 
    int(config['default']['out_features']),
    bool(config['default']['enable_relu']),
    int(config['default']['ai_version']))

def get_next_ai_version():
  """
  Get the next available version number from the ai_version file.
  If the file doesn't exist, write '2' to the file an return '1'.
  """
  ini = AISnakeGameConfig()
  ai_version_file = ini.ai_version_file()
  ai_version_file = os.path.join(lib_dir, ai_version_file)
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
  print(f"AI version is {ai_version}")
  return ai_version
 
def get_sim_desc(ai_version):
  """
  Get the key/value pairs that desribe a simulation run from 
  the simulation description file (e.g. models/sim_desc_v38.txt)
  """
  ini = AISnakeGameConfig()
  sim_desc_basename = ini.sim_desc_basename()
  sim_desc_dir = ini.sim_desc_dir()
  sim_desc_file = sim_desc_basename + str(ai_version) + '.txt'
  sim_desc_file = os.path.join(sim_desc_dir, sim_desc_file)
  if not os.path.isfile(sim_desc_file):
    print(f"ERROR: Unable to find simulation description file ({sim_desc_file}), exiting")
    sys.exit(1)
  config = configparser.ConfigParser()
  config.read(sim_desc_file)
  return config