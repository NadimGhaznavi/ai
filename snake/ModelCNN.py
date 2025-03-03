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
        self.plot = None
        input_size = ini.get('input_size')
        hidden_size = ini.get('hidden_size')
        output_size = ini.get('output_size')

        self.conv_b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=81, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=3)
        )
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelCNN initialization:    [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = self.conv_b1(x)
        self.plot.set_image_1(x[len(x) - 1])
        print(x[0])
        #x = self.conv_b2(x)
        x = x.view(x.size(0), -1)  # Flatten (batch_size, channels*height*width)
        x = self.out(x)
        x = x[0]
        return x.unsqueeze(0)
    
    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
    
    def set_plot(self, plot):
        self.plot = plot
