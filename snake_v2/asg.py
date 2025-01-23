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
from SnakeGamePlots import MyPlot
from AISnakeGameUtils import get_new_model, get_sim_desc, get_next_ai_version

def print_game_summary(ai_version, agent, score, record, game):
  summary = 'Snake AI (v' + str(ai_version) + ') ' + \
    'Game' + '{:>5}'.format(agent.n_games) + ', ' + \
    'Score' + '{:>4}'.format(score) + ', ' + \
    'Highscore' + '{:>4}'.format(record) + ', ' + \
    'Time ' + '{:>6}'.format(game.elapsed_time) + 's'
  if agent.nu_algo.print_stats:
    nu_algo_injected = agent.nu_algo.get_injected()
    summary = summary + ', NuAlgo: score ' + str(agent.nu_algo.get_nu_score()) + \
    ', pool ' + '{:>2}'.format(agent.nu_algo.get_nu_value()) + \
    ', reset# ' + str(agent.nu_algo.get_nu_refill_count()) + \
    ', bad game# ' + '{:>2}'.format(agent.nu_algo.get_bad_game_count()+1) + \
    ', injected# ' + str(nu_algo_injected)
  if agent.epsilon_algo.get_print_stats():
    summary = summary + ', EpsilonAlgo: injected# {:>3}'.format(agent.epsilon_algo.get_injected()) + \
      ', epsilon {:>3}'.format(agent.epsilon_algo.get_epsilon())
  summary = summary + ' - ' + game.lose_reason
  print(summary)

def train(ai_version, new_sim_run, my_plot):
  """
  This is the AI Snake Game main training loop.
  """
  # Get a mew instance of the AI Snake Game
  game = AISnakeGame(ai_version)

  # Initialize the simulation metrics
  plot_scores = [] # Scores for each game
  plot_mean_scores = [] # Average scores over a rolling window
  plot_times = [] # Times for each game
  plot_mean_times = [] # Average times over a rolling window

  if new_sim_run:
    # This is a new simulation
    config = AISnakeGameConfig() # Get the settings from the AISnakeGame.ini
    model = Linear_QNet(config) # Get a new model
    agent = AIAgent(game, model, config, ai_version) # Get a new instance of the AI Agent
    game.set_agent(agent)
    game.reset()
    agent.save_model()
    agent.save_sim_desc()
    
  else:
    # A version was passed into this script
    old_ai_version = ai_version
    config = AISnakeGameConfig()
    config = config.load_config(old_ai_version)
    model = Linear_QNet(config)
    ai_version = get_next_ai_version()
    game = AISnakeGame(ai_version)
    agent = AIAgent(game, model, config, ai_version)
    game.set_agent(agent)
    game.reset()
    agent.ai_version = old_ai_version
    agent.load_checkpoint()
    agent.ai_version = ai_version
    agent.nu_algo.nu_score = config.get('nu_score')
    agent.nu_algo.nu_value = config.get('nu_value')
    agent.nu_algo.nu_bad_games = config.get('nu_bad_games')
    
  total_score = 0 # Score for the current game
  record = 0 # Best score
  game.set_agent(agent) # Pass the agent to the game
    
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
      # Add a new layer when a specific score is reached
      if agent.b1_score > 0 and score >= agent.b1_score:
        agent.b1_score = 0 # Make sure we don't add another layer
        agent.model.insert_layer(1)
      if agent.b2_score > 0 and score >= agent.b2_score:
        agent.b2_score = 0
        agent.model.insert_layer(2)
      if agent.b3_score > 0 and score >= agent.b3_score:
        agent.b3_score = 0
        agent.model.insert_layer(3)
        
      agent.epsilon_algo.played_game()
      agent.nu_algo.played_game(score)
      # Train long memory
      game.reset()
      # Number of games the agent has played
      agent.n_games += 1
      # Implement the max_games feature where the simulation ends when the number 
      # of games reaches the max_games threashold
      
      if agent.max_games != 0 and agent.n_games == agent.max_games:
        lose_reason = "Executed max_games value of " + str(agent.max_games)
        game.lose_reason = lose_reason
        agent.set_config('lose_reason', lose_reason)
        agent.save_sim_desc()
        print_game_summary(ai_version, agent, score, record, game)
        my_plot.save()
        game.quit_game()

      agent.train_long_memory()
      if score > record:
        # New highscore!!! YAY!
        record = score

        # Check if the model has dynamic dropout layers
        if agent.model.has_dynamic_dropout():
          if agent.model.dropout_min != 0:
            if score >= agent.model.dropout_min:
              # Turn dropout on
              agent.model.set_p_value(agent.model.dropout_p)
            elif score <= agent.model.dropout_max:
              # Turn dropout off
              agent.model.set_p_value(0.0)
            
        agent.nu_algo.new_highscore(record) # Pass the new highscore to NuAlgo
        agent.save_checkpoint() # Save the simulation state
        game.sim_high_score = record
        agent.save_highscore(record) # Update the highscore file
        agent.highscore = record
        if agent.max_score != 0 and score >= agent.max_score:
          agent.max_score_num_count += 1
        if agent.max_score != 0 and score >= agent.max_score:
          # Exit the simulation if a score of max_score is achieved
          lose_reason = "Achieved max_score value of " + str(agent.max_score)
            
          game.lose_reason = lose_reason
          agent.set_config('lose_reason', lose_reason)
          agent.save_sim_desc()
          print_game_summary(ai_version, agent, score, record, game)
          my_plot.save()
          game.quit_game()
        #" a total of " + str(agent.max_score_num_count) + " times"

      print_game_summary(ai_version, agent, score, record, game)

      plot_scores.append(score)
      total_score += score
      mean_score = round(total_score / agent.n_games, 2)
      plot_mean_scores.append(mean_score)
      plot_times.append(game.elapsed_time)
      mean_time = round(game.sim_time / agent.n_games, 1)
      plot_mean_times.append(mean_time)

      
      my_plot.plot(plot_scores, plot_mean_scores, plot_times, plot_mean_times, ai_version)

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
  
  my_plot = MyPlot(ai_version)
  train(ai_version, new_sim_run, my_plot)
  




    
