"""
SimLogger.py

This file contains a class that can be used to log messages.
"""
import os

class SimLogger:

  def __init__(self, ini):
    self.ini = ini
    self.log_handle = None
    self.log('SimLogger initialization:   [OK]')

  def __del__(self):
    if self.log_handle:
      self.log_handle.close()

  def log(self, message):
    if not self.log_handle:
      data_dir = self.ini.get('sim_data_dir')
      sim_num = str(self.ini.get('sim_num'))
      log_basename = self.ini.get('log_basename')
      log_file = os.path.join(data_dir, sim_num + log_basename)
      print("Creating log file: " + log_file)
      self.log_handle = open(log_file, mode='w')
    self.log_handle.write(message + '\n')
    self.log_handle.flush()
    print(message)