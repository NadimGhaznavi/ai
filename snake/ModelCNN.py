"""
AISnakeGameCNN.py

This file contains a sub-class of nn.Module that is used to house the 
model that the AI agent uses when playing the Snake Game. 
"""
import torch
import torch.nn as nn
import os, sys

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig

class LinearQNet(nn.Module):
    def __init__(self, ini, log, model_level):
        super(LinearQNet, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.model_level = model_level

    def forward(self, x):
        pass
        