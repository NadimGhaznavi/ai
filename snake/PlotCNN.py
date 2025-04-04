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
        
        self.rows = 8 #ini.get('cnn_b3_channels') // 4
        self.cols = 8 #ini.get('cnn_b3_channels') // rows

        self.feature_maps = None  # To store feature maps

        # Set up the figure and axes for plotting feature maps
        self.fig, self.axs = plt.subplots(self.rows, self.cols, figsize=(5, 8), layout="tight", facecolor="#000000")
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
            x = self.cnn_model.upsample(x)
            x = self.cnn_model.conv_1(x)  # After first conv
            x = self.cnn_model.conv_2(x)
            x = self.cnn_model.conv_3(x)

        self.feature_maps_1 = x.squeeze(0)
        #self.feature_maps_2 = y.squeeze(0)  # Remove the batch dimension
        num_feature_maps_1 = self.feature_maps_1.shape[0]  # Should be 32
        #num_feature_maps_2 = self.feature_maps_2.shape[0]
        
        # Clear axes before plotting new feature maps
        for ax in self.axs.flat:
            ax.cla()
        
        print("DEBUG 1: ", num_feature_maps_1)
        #print("DEBUG 2: ", num_feature_maps_2)
        # Plot the feature maps
        for i in range(num_feature_maps_1):
            ax = self.axs[i // 8, i % self.cols]  # Arrange in a grid (8x4)
            #ax = self.axs[i // 3, i % 4]  # Arrange in a grid (8x4)
            feature_map = self.feature_maps_1[i].cpu().detach().numpy()
            #ax.imshow(feature_map, cmap='gray')
            ax.imshow(feature_map)
            ax.set_title(f'Feature Map {i + 1}')
            ax.axis('off')  # Turn off axis
        #for i in range(num_feature_maps_2):
        #    ax = self.axs[2 + (i // 4), i % self.cols]  # Arrange in a grid (8x4)
            #ax = self.axs[i // 3, i % 4]  # Arrange in a grid (8x4)
        #    feature_map = self.feature_maps_2[i].cpu().detach().numpy()
            #ax.imshow(feature_map, cmap='gray')
        #    ax.imshow(feature_map)
        #    ax.set_title(f'Feature Map {i + 1}')
        #    ax.axis('off')  # Turn off axis
    

        plt.show()
        plt.pause(0.1)  # Pause to update the figure
        plt.draw()

    def save(self):
        # Save the current figure if needed
        plot_file = 'feature_maps.png'  # Modify path as needed
        plt.savefig(plot_file)

