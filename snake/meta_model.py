import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

MODEL_DIR = './models'
MODEL_FILE = 'ai_model_v'
MODEL_FILE_SUFFIX = 'pth'

class MyReLU():
    def __init__(self, in_nodes, out_nodes, ai_ver):
        self.block = nn.Sequential()