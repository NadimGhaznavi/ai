# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelRNNT(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelRNNT, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        self.input_size = ini.get('rnnt_input_size')
        self.hidden_size = ini.get('rnnt_hidden_size')
        self.output_size = ini.get('output_size')
        self.rnn_layers = ini.get('rnn_layers')
        self.sequence_length = ini.get('rnnt_sequence_length')
        self.m_rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.rnn_layers)
        self.m_out = nn.Linear(self.hidden_size, self.output_size)
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelRNNT initialization:   [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        if len(x.size()) == 4:
            x = x[:, -1, :, :]
        h0 = torch.zeros(self.rnn_layers, x.size(1), self.hidden_size)
        out, _ = self.m_rnn(x, h0)
        out = out[:, -1, :]
        out = self.m_out(out)
        return out

    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)