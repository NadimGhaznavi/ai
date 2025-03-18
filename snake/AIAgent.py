"""
AIAgent.py
"""

import torch
from ModelCNN import ModelCNN
from ModelCNNR import ModelCNNR
from ModelCNNR3 import ModelCNNR3
from ModelCNNR4 import ModelCNNR4
from ModelL import ModelL
from ModelRNN import ModelRNN
from ReplayMemory import ReplayMemory
from EpsilonAlgo import EpsilonAlgo
from AITrainer import AITrainer
from NuAlgo import NuAlgo

class AIAgent:
    def __init__(self, ini, log, stats):
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        if ini.get('model') == 'linear':
            self.model = ModelL(ini, log, stats)
        elif ini.get('model') == 'rnn':
            self.model = ModelRNN(ini, log, stats)
        elif ini.get('model') == 'cnn':
            self.model = ModelCNN(ini, log, stats)
        elif ini.get('model') == 'cnnr':
            self.model = ModelCNNR(ini, log, stats)
        elif ini.get('model') == 'cnnr3':
            self.model = ModelCNNR3(ini, log, stats)
        elif ini.get('model') == 'cnnr4':
            self.model = ModelCNNR4(ini, log, stats)
        else:
            raise Exception(f"Unknown model type {ini.get('model')}")
        self.epsilon_algo = EpsilonAlgo(ini, log, stats)
        self.nu_algo = NuAlgo(ini, log, stats)
        self.memory = ReplayMemory(ini, log, stats)
        self.trainer = AITrainer(ini, log, stats, self.model)
        self.log.log('AIAgent initialization:     [OK]')
        self.last_dirs = [ 0, 0, 1, 0 ]

    def cleanup(self):
        self.stats.save()
        self.ini.save()

    def get_optimizer(self):
        return self.trainer.get_optimizer()

    def get_model(self):
        return self.model

    def get_move(self, state):
        random_move = self.epsilon_algo.get_move()
        if random_move:
            return random_move # Random move was returned
        random_move = self.nu_algo.get_move(self.stats.get('game', 'score'))
        if random_move:
            return random_move # Random move was returned
        # AI agent based action
        final_move = [0, 0, 0]
        if type(state) != torch.Tensor:
            state = torch.tensor(state, dtype=torch.float) # Convert to a tensor
        prediction = self.model(state) # Get the prediction
        move = torch.argmax(prediction).item() # Get the move
        final_move[move] = 1 # Set the move
        return final_move # Return
        
    def played_game(self, score):
        self.epsilon_algo.played_game()
        self.nu_algo.played_game(score)
        self.trainer.reset_steps()
        self.model.reset_steps()
        model_type = self.ini.get('model')
        if model_type == 'cnnr' or model_type == 'cnnr3' or model_type == 'cnnr4':
            self.model.reset_hidden()
        self.stats.set('agent', 'score', score)
 
    def remember(self, state, action, reward, next_state, done):
        # Store the state, action, reward, next_state, and done in memory
        if self.ini.get('enable_long_training'):
            self.memory.append((state, action, reward, next_state, done))

    def reset_epsilon_injected(self):
        self.epsilon_algo.reset_injected()

    def reset_nu_injected(self):
        self.nu_algo.reset_injected()

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.trainer.set_optimizer(optimizer)

    def train_long_memory(self):
        # Get the states, actions, rewards, next_states, and dones from the mini_sample
        enable = self.ini.get('enable_long_training')
        shuffle = self.ini.get('replay_mem_enable_shuffle')
        model_type = self.ini.get('model')
        if enable and shuffle:
            memory = self.memory.get_memory()
            for state, action, reward, next_state, done in memory:
                if model_type == 'cnn':
                    self.trainer.train_step_cnn(state, action, reward, next_state, done)
                else:
                    self.trainer.train_step(state, action, reward, next_state, done)
        else:
            num_games = 4
            while num_games > 0:
                num_games -= 1
                memory = self.memory.get_memory()
                if memory != False:                
                    if model_type == 'cnn' or model_type == 'cnnr' or model_type == 'cnnr3' or model_type == 'cnnr4':
                        for state, action, reward, next_state, done in memory[0]:
                            self.trainer.train_step_cnn(state, action, reward, next_state, [done])
                    else:
                        for state, action, reward, next_state, done in memory[0]:
                            self.trainer.train_step(state, action, reward, next_state, [done])

    def train_short_memory(self, state, action, reward, next_state, done):
        model_type = self.ini.get('model')
        if model_type == 'cnn' or model_type == 'cnnr' or model_type == 'cnnr3' or model_type == 'cnnr4':
            self.trainer.train_step_cnn(state, action, reward, next_state, [done])
        else:
            self.trainer.train_step(state, action, reward, next_state, [done])
