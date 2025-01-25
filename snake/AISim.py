"""
AISim.py

The frontend to the AI Snake Game.

"""
import os, sys
import matplotlib.pyplot as plt
import torch.nn as nn

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig
from AISnakeGame import AISnakeGame
from LinearQNet import LinearQNet
from AIAgent import AIAgent
from SnakeGamePlots import MyPlot

def print_game_summary(ini, agent, score, record, game, l2_score):
  ai_version = ini.get('ai_version')
  # Standard game summary metrics
  summary = 'Snake AI (v' + str(ai_version) + ') ' + \
    'Game' + '{:>5}'.format(agent.n_games) + ', ' + \
    'Score' + '{:>4}'.format(score) + ', ' + \
    'Highscore' + '{:>4}'.format(record) + ', ' + \
    'Time ' + '{:>6}'.format(game.elapsed_time) + 's'

  # Print the epsilon values
  if ini.get('epsilon_print_stats'):
    if score <= l2_score:
      if agent.l1_epsilon_algo.get_print_stats() and \
        agent.l1_epsilon_algo.get_epsilon() != 0:
        summary = summary + ', Model (1): inject# {:>3}'.format(agent.l1_epsilon_algo.get_injected()) + \
          ', pool ({:>3})'.format(agent.l1_epsilon_algo.get_epsilon())
    else:
      if agent.l2_epsilon_algo.get_print_stats() and \
        agent.l2_epsilon_algo.get_epsilon() != 0:
        summary = summary + ', Model(2): inject# {:>3}'.format(agent.l2_epsilon_algo.get_injected()) + \
          ', pool ({:>3})'.format(agent.l2_epsilon_algo.get_epsilon())

  # Level 1 and 2 statistics
  if ini.get('steps_stats'):
    l1_model_steps = agent.l1_model.get_steps()
    l2_model_steps = agent.l2_model.get_steps()
    l1_trainer_steps = agent.l1_trainer.get_steps()
    l2_trainer_steps = agent.l2_trainer.get_steps()
    summary = summary + \
      ', L1 steps: model# {:>5}'.format(l1_model_steps) + \
      ', trainer# {:>5}'.format(l1_trainer_steps) + \
      ', L2 steps: model# {:>5}'.format(l2_model_steps) + \
      ', trainer# {:>5}'.format(l2_trainer_steps)
    
  if ini.get('steps_stats') or ini.get('steps_verbose'):
    agent.l1_model.reset_steps()
    agent.l2_model.reset_steps()
    agent.l1_trainer.reset_steps()
    agent.l2_trainer.reset_steps()

  # Print the lose reason
  summary = summary + ' - ' + game.lose_reason
  print(summary)

def train():
  """
  This is the AI Snake Game main training loop.
  """
  # Get the AI Snake Game configuration
  ini = AISnakeGameConfig()
  ai_version = ini.get('ai_version')

  # Get our Matplotlib object
  my_plot = MyPlot(ini)

  # Get a mew instance of the AI Snake Game
  game = AISnakeGame(ini, my_plot)

  # Get a new instance of the AI Agent
  agent = AIAgent(ini, game) # Get a new instance of the AI Agent

  # Pass the agent to the game, we have to do this after instantiating
  # the game and the agent so that we avoid a circular reference
  game.set_agent(agent) # Pass the agent to the game

  if ini.get('new_simulation'):
    # This is a new simulation
    agent.save_model()
    agent.save_highscore(0)
    agent.ini.save_sim_desc()
    
  else:
    # A version was passed into this script
    agent.load_checkpoint()

  # Reset the AI Snake Game
  game.reset()

  # Initialize game metrics
  total_score = 0 # Score for the current game
  record = 0 # Best score

  # Initialize the matplotlib metrics
  plot_scores = [] # Scores for each game
  plot_mean_scores = [] # Average scores over a rolling window
  plot_times = [] # Times for each game
  plot_mean_times = [] # Average times over a rolling window

  # This is the score when we switch to using the level two network
  l2_score = ini.get('l2_score')

  print(f"AI version is {ai_version}")
  print(f"The second neural network will be used for scores above {ini.get('l2_score')}")

  ## The actual training loop
  while True:
    if ini.get('steps_verbose'):
      l1_model_steps = agent.l1_model.get_steps()
      l2_model_steps = agent.l2_model.get_steps()
      l1_trainer_steps = agent.l1_trainer.get_steps()
      l2_trainer_steps = agent.l2_trainer.get_steps()
      steps_summary = 'Steps: ' + \
        'L1 steps: model# {:>3}'.format(l1_model_steps) + ' trainer# {:>3}'.format(l1_trainer_steps) + \
        ', L2 steps: model# {:>3}'.format(l2_model_steps) + ' trainer# {:>3}'.format(l2_trainer_steps)
      print(steps_summary)
      
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

    # If the game is over
    if done:
      ## Dynamically add layers
      # Add a new layer when a specific score is reached
      if ini.get('b1_score') > 0 and score >= ini.get('b1_score'):
        ini.set_value('b1_score', 0) # Make sure we don't add another layer
        agent.l1_model.insert_layer(1)
      if ini.get('b2_score') > 0 and score >= ini.get('b2_score'):
        ini.set_value('b2_score', 0)
        agent.l1_model.insert_layer(2)
      if ini.get('b3_score') > 0 and score >= ini.get('b3_score'):
        ini.set_value('b3_score', 0)
        agent.l1_model.insert_layer(3)
      # Add a new layer when a specific score is reached
      if ini.get('l2_b1_score') > 0 and score >= ini.get('l2_b1_score'):
        ini.set_value('l2_b1_score', 0) # Make sure we don't add another layer
        agent.l2_model.insert_layer(1)
      if ini.get('l2_b2_score') > 0 and score >= ini.get('l2_b2_score'):
        ini.set_value('l2_b2_score', 0)
        agent.l2_model.insert_layer(2)
      if ini.get('l2_b3_score') > 0 and score >= ini.get('l2_b3_score'):
        ini.set_value('l2_b3_score', 0)
        agent.l2_model.insert_layer(3)
      
      if game.score <= l2_score:
        agent.l1_epsilon_algo.played_game()
      else:
        agent.l2_epsilon_algo.played_game()
      
      # Train long memory
      game.reset()
      # Number of games the agent has played
      agent.n_games += 1
      # Implement the max_games feature where the simulation ends when the number 
      # of games reaches the max_games threashold
      if ini.get('max_games') != 0 and agent.n_games == ini.get('max_games'):
        lose_reason = "Executed max_games value of " + str(ini.get('max_games'))
        game.lose_reason = lose_reason
        ini.set_value('lose_reason', lose_reason)
        agent.ini.save_sim_desc()
        print_game_summary(ini, agent, score, record, game, l2_score)
        my_plot.save()
        game.quit_game()

      agent.train_long_memory()
      if score > record:
        # New highscore!!! YAY!
        record = score
        # Save a checkpoint of the current AI model
        ini.set_value('sim_checkpoint_basename', f'checkpoint_l1_score_{record}.ptc')
        ini.set_value('l2_sim_checkpoint_basename', f'checkpoint_l2_score_{record}.ptc')
        agent.save_checkpoint()
        # Check if the model has dynamic dropout layers
        if agent.l1_model.has_dynamic_dropout():
          if agent.l1_model.dropout_min != 0:
            if score >= agent.l1_model.dropout_min:
              # Turn dropout on
              agent.l1_model.set_p_value(agent.l1_model.dropout_p)
            elif score <= agent.l1_model.dropout_max:
              # Turn dropout off
              agent.l1_model.set_p_value(0.0)
        # Check if the level 2 model has dynamic dropout layers
        if agent.l2_model.has_dynamic_dropout():
          if agent.l2_model.dropout_min != 0:
            if score >= agent.l2_model.dropout_min:
              # Turn dropout on
              agent.l2_model.set_p_value(agent.l2_model.dropout_p)
            elif score <= agent.l2_model.dropout_max:
              # Turn dropout off
              agent.l2_model.set_p_value(0.0)
            
        agent.save_checkpoint() # Save the simulation state
        game.sim_high_score = record
        agent.save_highscore(record) # Update the highscore file
        agent.highscore = record
        
        if ini.get('max_score') != 0 and score >= ini.get('max_score'):
          # Exit the simulation if a score of max_score is achieved
          lose_reason = "Achieved max_score value of " + str(ini.get('max_score'))
          game.lose_reason = lose_reason
          ini.set_value('lose_reason', lose_reason) 
          agent.ini.save_sim_desc()
          print_game_summary(ini, agent, score, record, game, l2_score)
          my_plot.save()
          game.quit_game()

      print_game_summary(ini, agent, score, record, game, l2_score)
      plot_scores.append(score)
      total_score += score
      mean_score = round(total_score / agent.n_games, 2)
      plot_mean_scores.append(mean_score)
      plot_times.append(game.elapsed_time)
      mean_time = round(game.sim_time / agent.n_games, 1)
      plot_mean_times.append(mean_time)

      
      my_plot.plot(plot_scores, plot_mean_scores, plot_times, plot_mean_times)

if __name__ == '__main__':
  train()
  




    
