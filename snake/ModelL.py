# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelL(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelL, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        input_size = ini.get('linear_input_size')
        hidden_size = ini.get('hidden_size')
        output_size = ini.get('output_size')
        main_block = nn.Sequential()
        main_block.append(nn.Linear(input_size, hidden_size))
        main_block.append(nn.ReLU())
        main_block.append(nn.Linear(hidden_size, hidden_size))
        main_block.append(nn.Linear(hidden_size, output_size))
        self.main_block = main_block
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelL initialization:      [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = self.main_block(x)
        return x
    
    def get_steps(self):
        return self.stats.get('model', 'steps')
    