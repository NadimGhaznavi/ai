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

        # A channel for the snake head, body and food
        input_channels = 3
        self.conv_layers = nn.Sequential(
            # First conv block: maintains spatial dimensions with padding.
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduces 20x20 -> 10x10

            # Second conv block:
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Reduces 10x10 -> 5x5
        )
        # The flattened feature size is 32 channels * 5 * 5 = 800.
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelCNN initialization:    [OK]")
    
    def forward(self, x):
        self.stats.incr('model', 'steps')
        #print("DEBUG 1 x.shape: ", x.shape)
        x = x.unsqueeze(0)  # Shape becomes [1, 3, 20, 20]
        #print("DEBUG 2 x.shape: ", x.shape)
        x = self.conv_layers(x)
        #print("DEBUG 3 x.shape: ", x.shape)
        # Optional visualization of feature maps
        if self.plot is not None:
            # Visualize the feature map from the first sample in the batch
            self.plot.set_image_2(x[0].detach().cpu())
        x = self.fc_layers(x)
        #print("DEBUG 4 x.shape: ", x.shape)
        return x

    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
    
    def set_plot(self, plot):
        self.plot = plot
