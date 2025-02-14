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
        #torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(ini.get('random_seed'))

    def reset_steps(self):
        self.stats.set('trainer', 'steps', 0)

    def train_step(self, state, action, reward, next_state, game_over):
        self.stats.incr('trainer', 'steps')
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        if self.ini.get('model') == 'linear' and len(state.shape) == 1:
            # Add a batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        elif self.ini.get('model') == 'rnn' and len(state.shape) == 1:
            # Add a batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        pred = self.model(state)
        target = pred.clone()

        if self.ini.get('model') == 'cnn' or self.ini.get('model') == 'cnnr':
            #print("DEBUG len(state.shape): ", len(state.shape))
            pred = self.model(state)
            target = pred.clone()
            if len(state.shape) == 3:
                state = torch.unsqueeze(state, 0)
                next_state = torch.unsqueeze(next_state, 0)
                action = torch.unsqueeze(action, 0)
                reward = torch.unsqueeze(reward, 0)
                game_over = (game_over, )
            pred = torch.unsqueeze(pred, 0)
            target = torch.unsqueeze(target, 0)

        #print("DEBUG state.shape: ", state.shape)
        #print("DEBUG next_state.shape: ", next_state.shape)
        #print("DEBUG action.shape: ", action.shape)
        #print("DEBUG reward.shape: ", reward.shape)
        #print("DEBUG game_over: ", game_over)   
        #print("DEBUG target.shape: ", target.shape)
        #print("DEBUG target: ", target)
        #print("DEBUG target[0]: ", target[0])


        for idx in range(len(game_over)):
            #print("DEBUG idx: ", idx)
            Q_new = reward[idx]
            if not game_over[idx]:
                #print('DEBUG next_state[idx].shape: ', next_state[idx].shape)
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad() # Reset the gradients to zero
        loss = self.criterion(target, pred) # Calculate the loss
        self.stats.set('trainer', 'loss', loss.item())
        loss.backward()
        self.optimizer.step() # Adjust the weights

            
            