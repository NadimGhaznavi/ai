# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class ModelCNNR(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelCNNR, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        self.plot = None

        c1_out_chan = ini.get('cnn_b1_channels')
        c2_out_chan = ini.get('cnn_b2_channels') # 32

        # Add an upsampling layer to increase input resolution from 20x20 to 40x40.
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

        # A channel for the snake head, body and food
        input_channels = 3
        self.conv_1 = nn.Sequential(
            # First conv block: maintains spatial dimensions with padding.
            nn.Conv2d(in_channels=input_channels, out_channels=c1_out_chan, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduces 40x40 -> 20x20
        )
        self.conv_2 = nn.Sequential(
            # Second conv block:
            nn.Conv2d(in_channels=c1_out_chan, out_channels=c2_out_chan, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Reduces 20x10 -> 10x10
        )
        # The flattened feature size is 32 channels * 10 * 10 = 3200.
        flatten_size = c2_out_chan * 10 * 10
        self.fc_cnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, flatten_size // 4),
            nn.ReLU(),
            nn.Linear(flatten_size // 4, flatten_size // 8),
            nn.ReLU(),
            nn.Linear(flatten_size // 8, 128),
            nn.ReLU()
        )
         # Use an LSTM to process the sequence of CNN embeddings.
        self.hidden_size = ini.get('lstm_hidden_size')  # Example hidden size; tweak as needed.
        self.dropout = ini.get('lstm_dropout') # E.g. 0.2 Dropout layer on the outputs of each LSTM layer except the last layer
        self.num_layers = ini.get('lstm_layers') # E.g. 4
        self.seq_length = ini.get('lstm_seq_length') # E.g. 10
        self.embedding_buffer = deque(maxlen=self.seq_length)

        # The LSTM will take in the 128-dim embedding at each time step.
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)

        # --- Output Layer ---
        # Final fully connected layer to output Q-values or move probabilities.
        self.output_size = ini.get('output_size')  # e.g., number of possible moves.
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)
        # --- End Output Layer ---

        # Hidden state for the LSTM (to be maintained across time steps in a game)
        self.hidden = None
        self.stats.set('model', 'steps', 0)
        self.log.log("ModelCNNR initialization:   [OK]")
    
    def forward(self, x):
        self.stats.incr('model', 'steps')
        x = x.unsqueeze(0)  # shape [1, 3, 20, 20]
        x = self.upsample(x)  # now shape [1, 3, 40, 40]
        x = self.conv_1(x)    # shape becomes [1, c1_out_chan, 20, 20]
        x = self.conv_2(x)    # shape becomes [1, c2_out_chan, 10, 10]
        embedding = self.fc_cnn(x)  # shape: [1, 128]
        embedding = embedding.unsqueeze(1)  # shape: [1, 1, 128]
        self.embedding_buffer.append(embedding.squeeze().detach())
        # Initialize or detach the hidden state as needed:
        if self.hidden is None:
            # Initialize hidden state and cell state with zeros
            self.hidden = (torch.zeros(self.num_layers, 1, self.hidden_size),
                           torch.zeros(self.num_layers, 1, self.hidden_size))
        else:
            # Detach hidden state from previous graph to avoid backpropagating through entire history
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        # Stack to create a tensor of shape [1, T, 128]
        embedding_seq = torch.stack(list(self.embedding_buffer), dim=0).unsqueeze(0)
        # embedding_seq shape: [batch_size, sequence_length, input_size] where batch_size is 1.
        lstm_out, self.hidden = self.lstm(embedding_seq, self.hidden)
        lstm_out = lstm_out[:, -1, :]  # Take output from the last time step
        
        # Final output: shape [1, output_size]
        out = self.fc_out(lstm_out)
        return out

    def reset_hidden(self):
        # Call this method at the start of a new game to reset the LSTM's hidden state.
        # The LSTM should maintain temporal state across time steps within a game. At the end
        # of the game, this should be reset.
        self.hidden = None
        self.embedding_buffer = deque(maxlen=self.seq_length)
        
    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
    
    def set_plot(self, plot):
        self.plot = plot
