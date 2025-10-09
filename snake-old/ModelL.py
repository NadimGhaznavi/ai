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
        hidden_size = ini.get('linear_hidden_size')
        output_size = ini.get('output_size')
        p_value = ini.get('linear_dropout')
        self.input_block = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.hidden_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.dropout_block = nn.Dropout(p=p_value)
        self.output_block = nn.Linear(hidden_size, output_size)
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelL initialization:      [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = self.input_block(x)
        x = self.hidden_block(x)
        x = self.dropout_block(x)
        x = self.output_block(x)
        return x
    
    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)

    def set_plot(self, plot):
        self.plot = plot
    