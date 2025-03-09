# Imports
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelCNNR3(nn.Module):
    def __init__(self, ini, log, stats):
        super(ModelCNNR3, self).__init__()
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        self.plot = None

        c1_out_chan = ini.get('cnn_b1_channels') # 16
        c2_out_chan = ini.get('cnn_b2_channels') # 24
        c3_out_chan = ini.get('cnn_b3_channels') # 32


        # Add an upsampling layer to increase input resolution from 20x20 to 40x40.
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic')

        # A channel each for the snake head, body and food
        input_channels = 3
        self.conv_1 = nn.Sequential(
            # First conv block: maintains spatial dimensions with padding.
            nn.Conv2d(in_channels=input_channels, out_channels=c1_out_chan, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Reduces 80x80 -> 40x40
        )
        self.conv_2 = nn.Sequential(
            # Second conv block:
            nn.Conv2d(in_channels=c1_out_chan, out_channels=c2_out_chan, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Reduces 40x40 -> 20x20
        )
        self.conv_3 = nn.Sequential(
            # Second conv block:
            nn.Conv2d(in_channels=c2_out_chan, out_channels=c3_out_chan, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   # Reduces 20x20 -> 10x10
        )
        # The flattened feature size is 32 channels * 10 * 10 = 3200.
        fc_in = c3_out_chan * 10 * 10
        self.fc_cnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 128),
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
        x = x.unsqueeze(0)  # shape [1, 3, 20, 20]
        x = self.upsample(x)  # now shape [1, 3, 80, 80]
        x = self.conv_1(x)    # shape becomes [1, 16, 40, 40]
        x = self.conv_2(x)    # shape becomes [1, 24, 20, 20]
        x = self.conv_3(x)    # shape becomes [1, 32, 10, 10]
        #with torch.no_grad():
        #    self.plot.set_image_2(x.squeeze()[-1].unsqueeze(0))
        embedding = self.fc_cnn(x)  # shape: [1, 128]
        embedding = embedding.unsqueeze(1)  # shape: [1, 1, 128]
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
