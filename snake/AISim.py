from SimConfig import SimConfig
from SimLogger import SimLogger
from SimPlot import SimPlot
from SimStats import SimStats
from AIAgent import AIAgent
from AISnakeGame import AISnakeGame
import time
import os
import torch
import sys

def checkpoint(config, stats, agent):
    # Model filename
    sim_num = config.get('sim_num')
    sim_data_dir = config.get('sim_data_dir')
    model_basename = config.get('model_basename')
    model_file = os.path.join(sim_data_dir, str(sim_num) + model_basename)
    # Metadata
    epoch = stats.get('game', 'num_games')
    model = agent.get_model()
    optimizer = agent.get_optimizer()
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            }, model_file)

def cleanup(agent, plot):
    agent.cleanup()
    plot.save()
    print('Simulation complete. Results in ' + agent.ini.get('sim_data_dir') + '.')

def restart(config, stats, log, agent, sim_num):
    # Model filename
    data_dir = config.get('data_dir')
    sim_data_dir = os.path.join(data_dir, str(sim_num))
    model_basename = config.get('model_basename')
    model_file = os.path.join(sim_data_dir, str(sim_num) + model_basename)
    if os.path.isfile(model_file):
        model = agent.get_model()
        optimizer = agent.get_optimizer()
        checkpoint = torch.load(model_file, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()
        agent.set_model(model)
        agent.set_optimizer(optimizer)
        stats.set('game', 'num_games', checkpoint['epoch'])
        log.log('Restored model from ' + model_file)
    else:
        log.log('No model file found: ' + model_file)
        sys.exit(1)

def train():
    """
    Train the AI agent to play the snake game using reinforcement learning.

    Initializes configuration, logging, statistics, AI agent, and the game.
    Runs a training loop where the agent interacts with the game environment,
    makes decisions based on the current state, learns from the feedback, and
    stores experiences. Updates are made to the agent's short and long-term
    memory. The game resets when a game-over condition is met. Training
    stops when the maximum number of epochs is reached.

    Updates the high score if a new one is achieved and logs game statistics
    after each game.
    """

    config = SimConfig()
    log = SimLogger(config)
    stats = SimStats(config, log)
    plot = SimPlot(config, log, stats)
    agent = AIAgent(config, log, stats)
    game = AISnakeGame(config, log, stats)
    
    if config.get('restart'):
        # Restart the simulation
        restart(config, stats, log, agent, config.get('restart'))
    
    model = agent.get_model()
    # For CNNs we want to render the game state
    if config.get('model') == 'cnnr' or config.get('model') == 'cnn' or config.get('model') == 'linear':
        model.set_plot(plot)
        game.board.set_plot(plot)
    
    if config.get('model') == 'rnn':
        game.board.set_plot(plot)

    # So that we can print the model from the game
    game.set_model(model)

    game.reset() # Reset the game
    
    log.log("AISim initialization:       [OK]")
    in_progress = True
    while in_progress:
        # The actual training loop
        old_state = game.board.get_state() # Get the current state
        move = agent.get_move(old_state) # Get the next move
        reward, game_over, score = game.play_step(move) # Play the game step
        new_state = game.board.get_state() # Get the new state
        agent.train_short_memory(old_state, move, reward, new_state, game_over) # Train short memory
        agent.remember(old_state, move, reward, new_state, game_over) # Remember
        if game_over:
            max_epochs = int(config.get('max_epochs'))
            nu_max_epochs = int(config.get('nu_max_epochs'))
            num_games = int(stats.get('game', 'num_games'))
            if max_epochs > 0 and max_epochs == num_games:
                in_progress = False # Reached max epochs
                log.log("Reached max epochs (" + str(max_epochs) + "), exiting")
            if nu_max_epochs > 0 and nu_max_epochs == num_games:
                log.log("Reached Nu max epochs (" + str(nu_max_epochs) + "), disabling Nu")
                config.set('nu_enabled', False)
            # Track how often a specific score has been reached
            stats.incr('scores', score)
            # Track the scores for each game
            stats.append('scores', 'all', score)
            agent.train_long_memory()
            if score > stats.get('game', 'highscore'):
                # New highscore!!! YAY!
                stats.set('game', 'highscore', score)
            game.reset() # Reset the game
            print_stats(log, stats, agent, config) # Print some stats
            agent.played_game(score) # Update the agent
            if num_games % 20 == 0:
                plot.plot() # Plot some stats
            if num_games % config.get('checkpoint_freq') == 0:
                checkpoint(config, stats, agent)

    cleanup(agent, plot)

def print_stats(log, stats, agent, config):
    summary = ''
    summary += 'AISim (v' + str(agent.ini.get('sim_num')) + '): Game {:<4}'.format(stats.get('game', 'num_games'))
    summary += ' Score: {:>2}'.format(stats.get('game', 'last_score'))
    summary += ', Time(s): {:6.2f}'.format(stats.get('game', 'game_time'))
    summary += ', Highscore: {:>3}'.format(stats.get('game', 'highscore'))
    if config.get('trainer_stats'):
        summary += ', Trainer steps {:>4}'.format(stats.get('trainer', 'steps'))
    if config.get('model_stats'):
        summary += ', Model steps {:>4}'.format(stats.get('model', 'steps'))
    if config.get('epsilon_enabled'):
        summary += ', Epsilon: {}'.format(stats.get('epsilon', 'status'))
        agent.reset_epsilon_injected()
    if config.get('nu_enabled'):
        agent.nu_algo.update_status()
        summary += ', Nu: {}'.format(stats.get('nu', 'status'))
        agent.reset_nu_injected()
    summary += ' - {}'.format(stats.get('game', 'lose_reason'))
    log.log(summary)

if __name__ == "__main__":
    train()