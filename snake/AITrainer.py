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
        self.log.log('AITrainer initialization:   [OK]')
        if ini.get('model') == 'cnn':
            self.optimizer = optim.SGD(model.parameters(), lr=self.lr)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
            self.criterion = nn.MSELoss()
        self.stats.set('trainer', 'steps', 0)
        self.steps = 0
        self.total_steps = 0
        self.cur_loss = 0.0
        #torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(ini.get('random_seed'))

    def reset_steps(self):
        self.steps = 0

    def train_step(self, state, action, reward, next_state, game_over):
        self.steps += 1
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        if len(state.shape) == 1:
            # Add a batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )
        pred = self.model(state)
        batch = False
        if self.ini.get('model') == 'rnnt':
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            reward = torch.unsqueeze(reward, 0)
            action = torch.unsqueeze(action, 0)
            game_over = (game_over, )
            if len(pred) > 1:
                batch = True
                state = state[len(state) -1 ]
                pred = pred[len(pred) -1 ].unsqueeze(0)
                reward = reward[len(reward) -1 ]
                action = action[len(action) - 1]
                next_state = next_state[len(next_state) - 1]
                game_over = (True,)

        target = pred.clone()
        if batch:
            reward = reward[len(reward) - 1].unsqueeze(0)
            action = action[len(action) - 1].unsqueeze(0)

        for idx in range(len(game_over)):
            self.stats.incr('trainer', 'steps')
            self.stats.incr('trainer', 'total_steps')
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad() # Reset the gradients to zero
        loss = self.criterion(target, pred) # Calculate the loss
        self.stats.set('trainer', 'loss', loss.item())
        loss.backward()
        self.optimizer.step() # Adjust the weights

            
            