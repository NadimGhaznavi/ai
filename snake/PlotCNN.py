""" 
PlotCNN.py
"""
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from IPython.utils import io
import os, sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

class PlotCNN:
    def __init__(self, log, ini, cnn_model):
        self.log = log
        self.ini = ini
        self.cnn_model = cnn_model
        
        rows = 8 #ini.get('cnn_b3_channels') // 4
        cols = 4 #ini.get('cnn_b3_channels') // rows

        self.feature_maps = None  # To store feature maps

        # Set up the figure and axes for plotting feature maps
        self.fig, self.axs = plt.subplots(rows, cols, figsize=(5, 8), layout="tight", facecolor="#000000")
        self.fig.suptitle('Feature Maps of CNN Layers', color="#00FF00")
        plt.ion()
        self.log.log("PlotCNN initialization:     [OK]")

    def __del__(self):
        plt.close()

    def plot(self, input_image):
        # Run the input image through the CNN to get the feature maps
        x = torch.tensor(input_image)
        x = x.unsqueeze(0)
        with torch.no_grad():
            # Pass through the first two conv layers and capture the feature maps
            #x = self.cnn_model.upsample(x)
            x = self.cnn_model.conv_1(x)  # After first conv
            x = self.cnn_model.conv_2(x)
            x = self.cnn_model.conv_3(x)

        self.feature_maps = x.squeeze(0)  # Remove the batch dimension
        num_feature_maps = self.feature_maps.shape[0]  # Should be 32
        # Clear axes before plotting new feature maps
        for ax in self.axs.flat:
            ax.cla()
        
        # Plot the feature maps
        for i in range(num_feature_maps):
            ax = self.axs[i // 4, i % 4]  # Arrange in a grid (8x4)
            #ax = self.axs[i // 3, i % 4]  # Arrange in a grid (8x4)
            feature_map = self.feature_maps[i].cpu().detach().numpy()
            #ax.imshow(feature_map, cmap='gray')
            ax.imshow(feature_map)
            ax.set_title(f'Feature Map {i + 1}')
            ax.axis('off')  # Turn off axis

        plt.show()
        plt.pause(0.1)  # Pause to update the figure
        plt.draw()

    def save(self):
        # Save the current figure if needed
        plot_file = 'feature_maps.png'  # Modify path as needed
        plt.savefig(plot_file)

