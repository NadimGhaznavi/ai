import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import time
import sys

class AITrainer():

    def __init__(self, ini, log, stats, model):
        self.ini = ini
        self.log = log
        self.stats = stats
        self.lr = ini.get('learning_rate')
        self.gamma = ini.get('discount')
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        if ini.get('model') == 'cnn' or ini.get('model') == 'rnn':
            self.criterion = nn.SmoothL1Loss()
            #self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.MSELoss()
        self.stats.set('trainer', 'steps', 0)
        #torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(ini.get('random_seed'))
        self.log.log('AITrainer initialization:   [OK]')

    def get_optimizer(self):
        return self.optimizer

    def reset_steps(self):
        self.stats.set('trainer', 'steps', 0)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_step(self, state, action, reward, next_state, game_over):
        self.stats.incr('trainer', 'steps')
        model_type = self.ini.get('model')
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        if model_type == 'linear' and len(state.shape) == 1:
            # Add a batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        elif model_type == 'rnn' and len(state.shape) == 1:
            # Add a batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        if model_type == 't':
            pred = self.model(state).squeeze(0)
        else:
            pred = self.model(state)
        target = pred.clone()

        if model_type == 't':
            # Ensure reward is a tensor of shape [1] if it's scalar
            if reward.dim() == 0:
                reward = reward.unsqueeze(0)
            if isinstance(game_over, bool):
                game_over = [game_over]
            for idx in range(len(game_over)):  
                Q_new = reward[idx]  # Ensure reward is properly indexed
                if not game_over[idx]:
                    with torch.no_grad():
                        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state))
                target[0, action[idx].argmax().item()] = Q_new  # Update Q value
        else:
            for idx in range(len(game_over)):  # Loop for RNN/Linear models
                Q_new = reward[idx]
                if not game_over[idx]:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                target[idx][action[idx].argmax().item()] = Q_new  # Update Q value

        self.optimizer.zero_grad()  # Reset gradients

        loss = self.criterion(target, pred) # Calculate the loss
        self.stats.set('trainer', 'loss', loss.item())
        self.stats.append('recent', 'loss', loss.item())
        loss.backward()
        self.optimizer.step() # Adjust the weights

