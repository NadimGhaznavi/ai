"""
AISim.py

The frontend to the AI Snake Game.

Linear_QNet(
self._in_features, 
self._b1_nodes, self._b1_layers,
self._b2_nodes, self._b2_layers,
self._b3_nodes, self._b3_layers,
self._out_features,
self.ai_version)

"""
import os, sys
import matplotlib.pyplot as plt
import torch.nn as nn
from QTrainer import QTrainer
import configparser
import contextlib

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig
from AISnakeGame import AISnakeGame
from LinearQNet import Linear_QNet
from AIAgent import AIAgent
from SnakeGamePlots import plot
from AISnakeGameUtils import get_new_model, get_sim_desc, get_next_ai_version

def print_game_summary(ai_version, agent, score, record, game):
  print('Snake AI (v' + str(ai_version) + ') ' + \
    'Game' + '{:>4}'.format(agent.n_games) + ', ' + \
    'Score' + '{:>4}'.format(score) + ', ' + \
    'Record' + '{:>4}'.format(record) + ', ' + \
    'Time ' + '{:>7}'.format(game.elapsed_time) + 's' + \
    ' - ' + game.lose_reason)

def train(ai_version, new_sim_run):
  """
  This is the AI Snake Game main training loop.
  """
  ini = AISnakeGameConfig()

  # Get a mew instance of the AI Snake Game
  game = AISnakeGame(ai_version)
  # The number of elements in the state map
  in_features = ini.get('in_features')
  # The number of valid snake moves i.e. straight, left or right
  out_features = ini.get('out_features')

  # Initialize the simulation metrics
  plot_scores = [] # Scores for each game
  plot_mean_scores = [] # Average scores over a rolling window
  plot_times = [] # Times for each game
  plot_mean_times = [] # Average times over a rolling window

  if new_sim_run:
    # This is a new simulation
    config = configparser.ConfigParser()
    config['default'] = {
      'in_features': in_features,
      'b1n': ini.get('b1_nodes'),
      'b1l': ini.get('b1_layers'),
      'b2n': ini.get('b2_nodes'),
      'b2l': ini.get('b2_layers'),
      'b3n': ini.get('b3_nodes'),
      'b3l': ini.get('b3_layers'),
      'out_features': out_features,
      'enable_relu': ini.get('enable_relu'),
      'epsilon_value': ini.get('epsilon_value'),
      'initial_epsilon_value': ini.get('epsilon_value'),
      'ai_version': ai_version
    }
    # Get a new model
    model = get_new_model(config)
    # Get a new instance of the AI Agent
    agent = AIAgent(game, model, config, ai_version)
    game.set_agent(agent)
    game.reset()
    agent.save_model()
    agent.save_sim_desc()

  else:
    # A version was passed into this script
    config = get_sim_desc(ai_version)
    model = get_new_model(config)
    agent = AIAgent(game, model, config, ai_version)
    game.set_agent(agent)
    game.reset()
    agent.load_checkpoint()
    agent.n_games = int(config['default']['num_games'])
    agent.epsilon_value = int(config['default']['epsilon_value'])

  total_score = 0 # Score for the current game
  record = 0 # Best score
  game.set_agent(agent) # Pass the agent to the game

  # Generate a new matplotlib figure and spec
  fig = plt.figure(figsize=(12,4), layout="tight")
  spec = fig.add_gridspec(ncols=1, nrows=2)
      
  ## The actual training loop
  while True:
    # Get old state
    state_old = agent.get_state()
    # Get move
    final_move = agent.get_action(state_old)
    # Perform move and get new state
    reward, done, score = game.play_step(final_move)
    state_new = agent.get_state()
    # Train short memory
    agent.train_short_memory(state_old, final_move, reward, state_new, done)
    # Remember
    agent.remember(state_old, final_move, reward, state_new, done)
    if done:
      # Train long memory
      game.reset()
      
      agent.n_games += 1
      # Implement the max_games feature where the simulation ends when the number 
      # of games reaches the max_games threashold
      if agent.max_games != 0 and agent.n_games == agent.max_games:
        lose_reason = "Executed max_games value of " + str(agent.max_games)
        game.lose_reason = lose_reason
        agent.set_config('lose_reason', lose_reason)
        agent.save_sim_desc()
        print_game_summary(ai_version, agent, score, record, game)
        game.quit_game()

      agent.train_long_memory()
      if score > record:
        record = score
        agent.save_checkpoint()
        game.sim_high_score = record
        agent.save_highscore(record)
        agent.highscore = record
        if agent.max_score != 0 and score >= agent.max_score:
          agent.max_score_num_count += 1
        if agent.max_score != 0 and \
          score > agent.max_score and \
          agent.max_score_num_count >= agent.max_score_num:
          # Exit the simulation if a score of max_score is achieved
          lose_reason = "Achieved max_score value of " + str(agent.max_score) + \
            " " + str(agent.max_score_num_count) + " times"
          game.lose_reason = lose_reason
          agent.set_config('lose_reason', lose_reason)
          agent.save_sim_desc()
          print_game_summary(ai_version, agent, score, record, game)
          game.quit_game()

      print_game_summary(ai_version, agent, score, record, game)

      plot_scores.append(score)
      total_score += score
      mean_score = round(total_score / agent.n_games, 2)
      plot_mean_scores.append(mean_score)
      plot_times.append(game.elapsed_time)
      mean_time = round(game.sim_time / agent.n_games, 1)
      plot_mean_times.append(mean_time)

      plot(plot_scores, plot_mean_scores, plot_times, plot_mean_times, ai_version)
      #plot2(fig, spec, plot_scores, plot_mean_scores, plot_times, plot_mean_times, AI_VERSION)

if __name__ == '__main__':
  ini = AISnakeGameConfig()
  # Get the ai_version from a command line switch
  ai_version = ini.get('ai_version')
  new_sim_run = True
  if ai_version:
    new_sim_run = False
  else:
    ### New AI Snake Game simulation...
    b1_nodes = ini.get('b1_nodes')
    if not b1_nodes:
      print("ERROR: You need to provide the number of nodes for the initial Linear layer:")
      print("  Use the -b1n switch to do this")
      sys.exit(1)
    else:
      # Get a new AI version for this simulation
      ai_version = get_next_ai_version()
  
  train(ai_version, new_sim_run)



    