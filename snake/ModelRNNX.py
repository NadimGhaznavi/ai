# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelRNNX(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelRNNX, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        input_size = ini.get('input_size')
        hidden_size = ini.get('hidden_size')
        output_size = ini.get('output_size')
        rnn_layers = ini.get('rnn_layers')
        self.m_in = nn.Linear(input_size, hidden_size)
        self.m_rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=rnn_layers)
        self.m_out = nn.Linear(hidden_size, output_size)
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelRNNX initialization:   [OK]")

    def forward(self, x):
        #print("DEBUG self.x_count: ", self.x_count)
        x = F.relu(self.m_in(x))
        # Parameters are: x.view(batch_size, sequence_length, input_size)
        #inputs = x.view(1, -1, self.ini.get('hidden_size'))
        inputs = x.view(1, -1, self.ini.get('hidden_size'))
        #print("DEBUG inputs.shape: ", inputs.shape)
        x, h_n = self.m_rnn(inputs)
        x = self.m_out(x)
        #print("DEBUG x.shape: ", x.shape)
        #return x[len(x) - 1]
        return x[0]

    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
        