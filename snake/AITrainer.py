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
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        #print("DEBUG state.shape: ", state.shape)
        #print("DEBUG len(state.shape): ", len(state.shape))
        #state = state.squeeze()
        #print("DEBUG after  state.shape: ", state.shape)
        #print("DEBUG: before game_over: ", game_over)
        #print("DEBUG: before type(game_over): ", type(game_over))

        if self.ini.get('model') == 'rnnt' and len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        #if self.ini.get('model') == 'rnnt' and len(state.shape) == 4:
        #    reward = reward[len(reward) -1 ].unsqueeze(0)
        #    state = state.squeeze()
        #    next_state = next_state.squeeze()
        #    action = action[len(action) - 1].unsqueeze(0)    

        #print("DEBUG reward: ", reward)
        #print("DEBUG reward.size(): ", reward.size())

        if len(state.shape) == 1:
            # Add a batch dimension
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )
        #print("DEBUG: after  game_over: ", game_over)
        #print("DEBUG: after  type(game_over): ", type(game_over))
        # 1. predicted Q values with current state
        pred = self.model(state)

        # 2. Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        target = pred.clone()
        #print("DEBUG len(game_over): ", len(game_over))
        #print("DEBUG target: ", target)
        for idx in range(len(game_over)):
            #print("DEBUG idx: ", idx)
            # Track the number of steps executed by this instance
            self.stats.incr('trainer', 'steps')
            self.stats.incr('trainer', 'total_steps')
            #print("DEBUG reward: ", reward)
            Q_new = reward[idx]
            #print("DEBUG target[idx]: ", target[idx])
            #print("DEBUG Q_new: ", Q_new)
            #print("DEBUG action: ", action)
            #print("DEBUG target: ", target)
            #print("DEBUG target[idx]: ", target[idx])
            #print("DEBUG torch.argmax(action).item(): ", torch.argmax(action).item())
            #print("DEBUG target[idx][torch.argmax(action).item()]: ", target[idx][torch.argmax(action).item()])
            
            if not game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                #print("DEBUG Q_new: ", Q_new)
            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad() # Reset the gradients to zero
        loss = self.criterion(target, pred) # Calculate the loss
        self.stats.set('trainer', 'loss', loss.item())
        #loss.backward(retain_graph=True) # Backpropagate the loss
        loss.backward()
        self.optimizer.step() # Adjust the weights

            
            