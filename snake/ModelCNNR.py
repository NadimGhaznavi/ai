# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelCNNR(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelCNNR, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        self.plot = None
        input_size = ini.get('input_size')
        hidden_size = ini.get('hidden_size')
        output_size = ini.get('output_size')

        self.conv_b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            #nn.Conv2d(in_channels=10, out_channels=10, kernel_size=2, stride=1, padding=0),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_b2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.rnn = nn.RNN(input_size=9, hidden_size=9, num_layers=2)
        self.out = nn.Sequential(
            nn.Flatten(),
            #nn.ReLU(),
            #nn.Linear(in_features=9, out_features=9),
            nn.ReLU(),
            nn.Linear(in_features=9, out_features=3)
        )
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelCNNR initialization:   [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = self.conv_b1(x)
        #x = self.conv_b2(x)
        pic_1 = x[len(x) - 1]
        #print("DEBUG pic_1: ", pic_1)
        self.plot.set_image_1(pic_1)
        inputs = x.view(1, -1, 9)
        x, h_n = self.rnn(inputs)
        x = self.out(x[len(x) - 1])
        x = x[0]
        return x
    
    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
    
    def set_plot(self, plot):
        self.plot = plot
