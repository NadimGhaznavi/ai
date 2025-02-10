"""
AISim.py

The frontend to the AI Snake Game.

"""
import os, sys
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig
from AISnakeGame import AISnakeGame
from LinearQNet import LinearQNet
from AIAgentX import AIAgent
#from AIAgentN import AIAgent
from SnakeGamePlots import MyPlot
from AILogger import AILogger
from ReplayMemory import ReplayMemory
from QTrainer import QTrainer
from collections import deque

MIN_SCORE = 2
MAX_DATA_POINTS = 1000
def print_game_summary(ini, log, agent, score, record, game, stats):
  ai_version = ini.get('ai_version')
  # Standard game summary metrics
  summary = 'Snake AI (v' + str(ai_version) + ') ' + \
    'Game' + '{:>5}'.format(game.get_game_num()) + ', ' + \
    'Score' + '{:>4}'.format(score) + ', ' + \
    'Highscore' + '{:>4}'.format(record) + ', ' + \
    'Time ' + '{:>6}'.format(game.elapsed_time) + 's'

  # Print the epsilon values
  if ini.get('epsilon_print_stats') and agent.get_epsilon_value():
    summary = summary + ', {}'.format(agent.get_epsilon())
    agent.reset_epsilon_injected()

  # Print the nu values
  if ini.get('nu_print_stats') and agent.nu_algo.is_enabled():
    summary = summary + ', {}'.format(agent.get_nu_algo())
    agent.reset_nu_algo_injected()

  # Model and trainer steps
  if ini.get('steps_stats'):
    summary = summary + ', ' + agent.get_model_steps(score)
    summary = summary + ', ' + agent.get_trainer_steps(score)
  if ini.get('steps_stats') or ini.get('steps_verbose'):
    agent.reset_model_steps()
    agent.reset_trainer_steps()

  if ini.get('steps_stats_all'):
    # All model and trainer steps and lose reason
    summary = summary + ' - ' + game.lose_reason + '\n'
    summary = summary + agent.get_all_steps()
  else:
    # Lose reason
    summary = summary + ' - ' + game.lose_reason

  agent.reset_model_steps()
  agent.reset_trainer_steps()

  log.log(summary)
  agent.log_scores()
  agent.log_upgrades(stats)
  agent.log_loss()

def restart_simulation(agent, ini, log):
  """
  Restart a simulation. This involves checking for checkpoint files and
  loading them if they exist.
  """
  data_dir = ini.get('sim_data_dir')
  checkpoint_basename = ini.get('sim_checkpoint_basename')
  for level in range(0, 99):
    checkpoint_file = os.path.join(lib_dir, os.path.join(data_dir, str(ini.get('restart')) + '_L' + str(level) + checkpoint_basename))
    print("Checking for checkpoint file (" + checkpoint_file, end='')
    if os.path.isfile(checkpoint_file):
      print(") [OK]")
      model = LinearQNet(ini, log, level)
      trainer = QTrainer(ini, model, level)
      optimizer = trainer.optimizer
      agent.level[level] = {}
      agent.level[level]['model'] = model
      agent.level[level]['memory'] = ReplayMemory(ini)
      agent.level[level]['trainer'] = trainer
      # Do the actual load
      print(f"Loading L{level} checkpoint file: " + checkpoint_file)
      checkpoint = torch.load(checkpoint_file, weights_only=False)
      state_dict = checkpoint['model_state_dict']
      model = agent.level[level]['model']
      model.load_state_dict(state_dict)
      optimizer = agent.level[level]['trainer'].optimizer
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
      print(") [NOT FOUND]")

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

  # Check if we are restarting a simulation
  if ini.get('restart'):
    restart_simulation(agent, ini, log)

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
  plot_scores = deque(maxlen=MAX_DATA_POINTS) # Scores for each game
  plot_mean_scores = deque(maxlen=MAX_DATA_POINTS) # Average scores over a rolling window
  plot_times = deque(maxlen=MAX_DATA_POINTS) # Times for each game
  plot_mean_times = deque(maxlen=MAX_DATA_POINTS) # Average times over a rolling window

  log.log(f"AI Snake Game simulation number is {ini.get('ai_version')}")
  log.log(f"Configuration file being used is {ini.get('ini_file')}")

  # Flag, indicating whether the L2 model was updated from L1
  L2_updated = False


  stats = {}
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
    if reward > 0:
      # It's a good reward, food!
      agent.share_dream(score)
    # Print verbose step stats
    if ini.get('steps_verbose'):
      print(agent.get_all_steps())

    # If the game is over
    if done:
      if score not in stats:
        stats[score] = { 
          'bad_games': 1, 
          'upgrades': 0 
          }
      else:
        stats[score]['bad_games'] += 1

      # Only upgrade these and higher
      if ini.get('agent_x_enable_promo'):
        if score > ini.get('agent_x_min_score') and stats[score]['bad_games'] >= 9:
          stats[score]['bad_games'] = 0
          stats[score]['upgrades'] += 1
          agent.set_stats(stats)
          agent.upgrade(score, stats[score]['upgrades'])

      # Disable the NuAlgo after game number 600
      nu_disable_games = ini.get('nu_disable_games')
      if nu_disable_games and game.get_num_games() > nu_disable_games:
        agent.nu_algo.disable()

      # Update the Agent with the new score (used by epsilon and NuAlgo)
      agent.played_game(score)

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

      print_game_summary(ini, log, agent, score, record, game, stats)
      plot_scores.append(score)
      total_score += score
      num_games = game.get_game_num()
      mean_score = round(total_score / num_games, 2)
      plot_mean_scores.append(mean_score)
      plot_times.append(game.elapsed_time)
      mean_time = round(game.sim_time / num_games, 1)
      plot_mean_times.append(mean_time)
      my_plot.plot(plot_scores, plot_mean_scores, plot_times, plot_mean_times)

if __name__ == '__main__':
  train()
  




    
