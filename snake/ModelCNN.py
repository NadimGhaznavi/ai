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
        input_size = ini.get('input_size')
        hidden_size = ini.get('hidden_size')
        output_size = ini.get('output_size')

        self.conv_b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=100, out_features=3)
        )
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelCNN initialization:    [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        #print("DEBUG 1 x.shape: ", x.shape)
        x = self.conv_b1(x)
        #print("DEBUG 2 x.shape: ", x.shape)
        #x = self.conv_b2(x)
        #print("DEBUG 3 x.shape: ", x.shape)
        #if len(x.size()) == 4:
            # Chop off the batch, just return the last one
        #    x = x[len(x) - 1]
        x = self.out(x)
        #print("DEBUG 4 x.shape: ", x.shape)
        x = x[0]
        #print("DEBUG x.shape: ", x.shape)
        #print("DEBUG x: ", x)
        return x
    
    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
    