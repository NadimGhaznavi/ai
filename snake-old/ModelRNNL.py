# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelRNNL(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelRNNL, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        input_size = ini.get('linear_input_size')
        l_hidden_size = ini.get('hidden_size')
        rnn_hidden_size = ini.get('rnn_hidden_size')
        output_size = ini.get('output_size')
        rnn_layers = ini.get('rnn_layers')
        rnn_dropout = ini.get('rnn_dropout')

        self.m_in = nn.Sequential()
        self.m_in.append(nn.Linear(input_size, l_hidden_size))
        self.m_in.append(nn.ReLU())
        self.m_in.append(nn.Linear(l_hidden_size, rnn_hidden_size))
        self.m_out = nn.Linear(l_hidden_size, output_size)
        self.m_rnn = nn.RNN(input_size=rnn_hidden_size, hidden_size=l_hidden_size, num_layers=rnn_layers, dropout=rnn_dropout)
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelRNN initialization:    [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = self.m_in(x)
        inputs = x.view(1, -1, self.ini.get('rnn_hidden_size'))
        x, h_n = self.m_rnn(inputs)
        x = self.m_out(x)
        return x[len(x) - 1]

    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
        