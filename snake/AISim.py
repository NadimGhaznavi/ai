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
from AILogger import AILogger

def print_game_summary(ini, log, agent, score, record, game):
  ai_version = ini.get('ai_version')
  # Standard game summary metrics
  summary = 'Snake AI (v' + str(ai_version) + ') ' + \
    'Game' + '{:>5}'.format(agent.n_games) + ', ' + \
    'Score' + '{:>4}'.format(score) + ', ' + \
    'Highscore' + '{:>4}'.format(record) + ', ' + \
    'Time ' + '{:>6}'.format(game.elapsed_time) + 's'

  # Print the epsilon values
  if ini.get('epsilon_print_stats') and agent.get_epsilon():
    summary = summary + ', {}'.format(agent.get_epsilon())
    agent.reset_epsilon_injected()

  # Print the nu values
  if ini.get('nu_print_stats'):
    summary = summary + ', {}'.format(agent.get_nu_algo())
    agent.reset_nu_algo_injected()

  # Model and trainer steps
  if ini.get('steps_stats'):
    summary = summary + ', ' + agent.get_model_steps(score)
    summary = summary + ', ' + agent.get_trainer_steps(score)
  if ini.get('steps_stats') or ini.get('steps_verbose'):
    agent.reset_model_steps(score)
    agent.reset_trainer_steps(score)

  if ini.get('steps_stats_all'):
    # All model and trainer steps and lose reason
    summary = summary + ' - ' + game.lose_reason + '\n'
    summary = summary + agent.get_all_steps()
  else:
    # Lose reason
    summary = summary + ' - ' + game.lose_reason

  agent.reset_model_steps(score)
  agent.reset_trainer_steps(score)


  log.log(summary)

def train():
  """
  This is the AI Snake Game main training loop.
  """
  # Get the AI Snake Game configuration
  ini = AISnakeGameConfig()

  # Get a logger object
  log = AILogger(ini)

  # Get our Matplotlib object
  my_plot = MyPlot(ini)

  # Get a mew instance of the AI Snake Game
  game = AISnakeGame(ini, log, my_plot)

  # Get a new instance of the AI Agent
  agent = AIAgent(ini, log, game) # Get a new instance of the AI Agent

  # Pass the agent to the game, we have to do this after instantiating
  # the game and the agent so that we avoid a circular reference
  game.set_agent(agent) # Pass the agent to the game

  # Initalize the highscore file
  agent.set_highscore(0)

  # Save the simulation configuration
  agent.ini.save_sim_desc()

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

  log.log(f"AI Snake Game simulation number is {ini.get('ai_version')}")
  log.log(f"Configuration file being used is {ini.get('ini_file')}")

  # Flag, indicating whether the L2 model was updated from L1
  L2_updated = False

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

    # Print verbose step stats
    if ini.get('steps_verbose'):
      print(agent.get_all_steps())

    # If the game is over
    if done:
      # Update the Agent with the new score (used by epsilon and NuAlgo)
      agent.played_game(score)

      # Copy the model's weights and bias' to the L2 model when the L2 score is reached
      agent.save_checkpoint()

      # Perform a checkpoint every 100 games
      if game.get_num_games() % 100 == 0:
        agent.save_checkpoint()
      
      # Train long memory
      game.reset()
      # Number of games the agent has played
      agent.increment_games()
      # Implement the max_games feature where the simulation ends when the number 
      # of games reaches the max_games threashold
      if ini.get('max_games') != 0 and agent.n_games == ini.get('max_games'):
        lose_reason = "Executed max_games value of " + str(ini.get('max_games'))
        game.set_lose_reason(lose_reason)
        ini.set_value('lose_reason', lose_reason)
        agent.ini.save_sim_desc()
        print_game_summary(ini, agent, score, record, game)
        my_plot.save()
        game.quit_game()

      agent.train_long_memory()
      if score > record:
        # New highscore!!! YAY!
        record = score

        # NuAlgo
        if ini.get('nu_enable'):
          agent.set_nu_algo_highscore(score)

        ## Save a checkpoint of the current AI model
        agent.save_checkpoint()
            
        game.set_highscore(record)
        agent.set_highscore(record)
        
        if ini.get('max_score') != 0 and score >= ini.get('max_score'):
          # Exit the simulation if a score of max_score is achieved
          lose_reason = "Achieved max_score value of " + str(ini.get('max_score'))
          game.set_lose_reason(lose_reason)
          ini.set_value('lose_reason', lose_reason) 
          agent.ini.save_sim_desc()
          print_game_summary(ini, agent, score, record, game)
          my_plot.save()
          game.quit_game()

      print_game_summary(ini, log, agent, score, record, game)
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
  




    
