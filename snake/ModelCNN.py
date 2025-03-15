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
        self.cnn_b1_channels = ini.get('cnn_b1_channels')
        self.cnn_b2_channels = ini.get('cnn_b2_channels')
        output_size = ini.get('output_size')

        # A channel for the snake head, body and food
        input_channels = 3
        self.conv_1 = nn.Sequential(
            # First conv block: maintains spatial dimensions with padding.
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=self.cnn_b1_channels, 
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))  # Reduces 20x20 -> 10x10
        self.conv_2 = nn.Sequential(
            # Second conv block:
            nn.Conv2d(in_channels=self.cnn_b1_channels, 
                      out_channels=self.cnn_b2_channels, 
                      kernel_size=3, 
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Reduces 10x10 -> 5x5
        )
        # The flattened feature size is 32 channels * 5 * 5 = 800.
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.cnn_b2_channels * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, output_size)
        )
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelCNN initialization:    [OK]")
    
    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = x.unsqueeze(0)  # Shape becomes [1, 3, 20, 20]
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.fc_layers(x)
        return x

    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
    
    def set_plot(self, plot):
        self.plot = plot
