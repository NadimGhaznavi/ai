# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelRNN(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelRNN, self).__init__()
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
        self.log.log("ModelL initialization:      [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = F.relu(self.m_in(x))
        inputs = x.view(1, -1, self.ini.get('hidden_size'))
        x, h_n = self.m_rnn(inputs)
        x = self.m_out(x)
        return x[len(x) - 1]
    
    def get_steps(self):
        return self.stats.get('model', 'steps')
    