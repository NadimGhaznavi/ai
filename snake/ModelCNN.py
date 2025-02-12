# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelCNN(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelCNN, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        input_size = ini.get('linear_input_size')
        hidden_size = ini.get('hidden_size')
        output_size = ini.get('output_size')
        # nn.Conv2d(color_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelCNN initialization:    [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_steps(self):
        return self.stats.get('model', 'steps')
    