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
        self.x = None
        self.x_count = 0
        self.log.log("ModelL initialization:      [OK]")

    def forward(self, x):
        self.stats.incr('model', 'steps')
        print("DEBUG self.x_count: ", self.x_count)
        x = F.relu(self.m_in(x))
        # Parameters are: x.view(batch_size, sequence_length, input_size)
        if self.x is None:
            self.x = x
            self.x_count += 1
        else:
            print("DEBUG before      x.size(): ", x.size())
            if x.dim() == 1:
                x = x.unsqueeze(0)
            print("DEBUG after       x.size(): ", x.size())
            print("DEBUG before self.x.size(): ", self.x.size())
            for row in x:
                self.x = torch.cat((self.x, row.unsqueeze(0)), 0)
                self.x_count += 1
        
            print("DEBUG after  self.x.size(): ", self.x.size())
            print("DEBUG     self.x.size(): ", self.x.size())
        
        #if self.x_count == 501:
        #    print("DEBUG before self.x.size(): ", self.x.size())
        #    self.x = self.x[1:]
        #    print("DEBUG after  self.x.size(): ", self.x.size())
        #    self.x_count = 500

        #inputs = self.x.view(1, -1, self.ini.get('hidden_size'))
        inputs = self.x.view(self.x_count, 1, self.ini.get('hidden_size'))
        #inputs = x.view(1, -1, self.ini.get('hidden_size'))
        print("DEBUG inputs.shape: ", inputs.shape)
        x, h_n = self.m_rnn(inputs)
        x = self.m_out(x)
        print("DEBUG x.shape: ", x.shape)
        return x[len(x) - 1]
    
    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_x(self):
        self.x = None
    