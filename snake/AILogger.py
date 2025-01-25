"""
AILogger.py

This file contains a class that can be used to log messages.
"""
import os

class AILogger:

  def __init__(self, ini):
    self.ini = ini
    self.log_file = None
    self.log_file = str(ini.get('ai_version')) + ini.get('log_basename')
    self.log_file = os.path.join(ini.get('sim_data_dir'), self.log_file)
    self.log_file = open(self.log_file, 'w')

  def __del__(self):
    if self.log_file:
      self.log_file.close()

  def log(self, message):
    self.log_file.write(message + '\n')
    self.log_file.flush()
    print(message)