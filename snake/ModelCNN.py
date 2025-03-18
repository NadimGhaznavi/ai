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
        self.cnn_b3_channels = ini.get('cnn_b3_channels')
        output_size = ini.get('output_size')
        self.enable_dropout = ini.get('cnn_enable_dropout')
        if self.enable_dropout:
            dropout = ini.get('cnn_dropout')

        #self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')
        # A channel for the snake head, body and food
        input_channels = 3
        padding_mode = 'zeros'
        padding = 0
        b1_kernel_size = 3
        b2_kernel_size = 3
        b3_kernel_size = 3
        self.conv_1 = nn.Sequential(
            # First conv block: maintains spatial dimensions with padding.
            nn.Conv2d(in_channels=input_channels, 
                      out_channels=self.cnn_b1_channels, 
                      kernel_size=b1_kernel_size, padding=padding, padding_mode=padding_mode),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)  # Reduces 20x20 -> 10x10
        )
        self.conv_2 = nn.Sequential(
            # Second conv block:
            nn.Conv2d(in_channels=self.cnn_b1_channels, 
                      out_channels=self.cnn_b2_channels, 
                      kernel_size=b2_kernel_size, padding=padding, padding_mode=padding_mode),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduces 20x20 -> 10x10
        )
        self.conv_3 = nn.Sequential(
            # Third conv block:
            nn.Conv2d(in_channels=self.cnn_b2_channels, 
                      out_channels=self.cnn_b3_channels, 
                      kernel_size=b3_kernel_size, padding=padding, padding_mode=padding_mode),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Reduces 20x20 -> 10x10
        )
        # The flattened feature size is 32 channels * 5 * 5 = 800.
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(288, 128),
            nn.ReLU()
        )
        # Dropout layer
        if self.enable_dropout:
            self.dropout_layer = nn.Dropout(p=dropout)
        # Output 
        self.output_layer = nn.Linear(128, output_size)
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelCNN initialization:    [OK]")
    
    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = x.unsqueeze(0)  # Shape becomes [1, 3, 20, 20]
        #x = self.upsample(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.fc_layer(x)
        if self.enable_dropout:
            x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x

    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
    
    def set_plot(self, plot):
        self.plot = plot
