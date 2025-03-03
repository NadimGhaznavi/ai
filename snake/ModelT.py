import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelPos import ModelPos

class ModelT(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelT, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        
        self.ini = ini
        self.log = log
        self.stats = stats
        
        input_size = ini.get('linear_input_size')
        hidden_size = ini.get('t_hidden_size')
        output_size = ini.get('output_size')
        num_heads = ini.get('t_num_heads')
        num_layers = ini.get('t_num_layers')
        feedforward_dim = ini.get('t_feedforward_dim')
        dropout = ini.get('t_dropout')
        
        self.embedding = nn.Linear(input_size, hidden_size)  # Project input to hidden size
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=feedforward_dim, 
                                                   dropout=dropout, 
                                                   batch_first=True)
        self.positional_encoder = ModelPos(hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelTransformer initialization: [OK]")
    
    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = self.embedding(x)  # Project input to hidden size
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        x = self.positional_encoder(x) # Add positional encoding
        x = self.transformer_encoder(x)  # Pass through Transformer
        x = self.fc_out(x)  # Final linear layer
        return x
    
    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)

    def set_plot(self, plot):
        self.plot = plot