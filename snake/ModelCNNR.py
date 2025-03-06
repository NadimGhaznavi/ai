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

        # A channel for the snake head, body and food
        input_channels = 3
        self.conv_layers = nn.Sequential(
            # First conv block: maintains spatial dimensions with padding.
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduces 20x20 -> 10x10

            # Second conv block:
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Reduces 10x10 -> 5x5
        )
        # The flattened feature size is 32 channels * 5 * 5 = 800.
        self.fc_cnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU()
        )
         # Use an LSTM to process the sequence of CNN embeddings.
        self.hidden_size = ini.get('cnnr_hidden_size')  # Example hidden size; tweak as needed.
        # The LSTM will take in the 128-dim embedding at each time step.
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        # --- End RNN Part ---

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
        x = x.unsqueeze(0)  # Shape becomes [1, 3, 20, 20]
        cnn_out = self.conv_layers(x) # CNN block
        # Expected shape [1, 32, 5, 5]
        # Obtain the CNN embedding: shape [1, 128]
        embedding = self.fc_cnn(cnn_out)
        # For the RNN, add a time dimension so that embedding is [batch, seq_len, feature_dim]
        # Here, each forward call represents one time step: [1, 1, 128]
        embedding = embedding.unsqueeze(1)
        # Initialize or detach the hidden state as needed:
        if self.hidden is None:
            # Initialize hidden state and cell state with zeros
            self.hidden = (torch.zeros(1, 1, self.hidden_size),
                           torch.zeros(1, 1, self.hidden_size))
        else:
            # Detach hidden state from previous graph to avoid backpropagating through entire history
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        # Pass through the LSTM
        lstm_out, self.hidden = self.lstm(embedding, self.hidden)
        # lstm_out has shape [1, 1, hidden_size]; take the output from the last time step
        lstm_out = lstm_out[:, -1, :]  # shape: [1, hidden_size]
        
        # Final output: shape [1, output_size]
        out = self.fc_out(lstm_out)
        return out

    def reset_hidden(self):
        # Call this method at the start of a new game to reset the LSTM's hidden state.
        self.hidden = None
        
    def get_steps(self):
        return self.stats.get('model', 'steps')
    
    def reset_steps(self):
        self.stats.set('model', 'steps', 0)
    
    def set_plot(self, plot):
        self.plot = plot
