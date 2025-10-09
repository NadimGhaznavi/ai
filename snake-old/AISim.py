from SimConfig import SimConfig
from SimLogger import SimLogger
from SimPlot import SimPlot
from SimStats import SimStats
from AIAgent import AIAgent
from AISnakeGame import AISnakeGame
from PlotCNN import PlotCNN
import os
import torch
import sys
import time

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

def save_model(config, model):
    data_dir = config.get('sim_data_dir')
    sim_num = str(config.get('sim_num'))
    model_basename = config.get('model_desc_basename')
    model_file = os.path.join(data_dir, sim_num + model_basename)
    print("Saving model to file: " + model_file)
    handle = open(model_file, mode='w')
    handle.write(str(model) + '\n')

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
    model_type = config.get('model')
    log = SimLogger(config)
    stats = SimStats(config, log)
    plot = SimPlot(config, log, stats)
    agent = AIAgent(config, log, stats)
    model = agent.get_model()
    model.set_plot(plot)
    save_model(config, model)
    game = AISnakeGame(config, log, stats)
    game.set_agent(agent)
    plot_cnn_enable = config.get('plot_cnn_enable')
    if plot_cnn_enable and model_type == 'cnnr' or model_type == 'cnn' or model_type == 'cnnr4':
        cnn_plot = PlotCNN(log, config, model)
        plot_cnn_freq = config.get('plot_cnn_freq')
    if config.get('restart'):
        # Restart the simulation
        restart(config, stats, log, agent, config.get('restart'))
    
    model = agent.get_model()
    # For CNNs we want to render the game state
    if model_type == 'cnnr' or model_type == 'cnn' or model_type == 'cnnr3' or \
        model_type == 'cnnr4' or model_type == 'linear':
        model.set_plot(plot)
        game.board.set_plot(plot)
    
    if model_type == 'rnn':
        game.board.set_plot(plot)

    # So that we can print the model from the game
    game.set_model(model)

    game.reset() # Reset the game
    
    log.log("AISim initialization:       [OK]")
    in_progress = True
    max_epochs = int(config.get('max_epochs'))
    enable_long_training = config.get('enable_long_training')
    nu_max_epochs = int(config.get('nu_max_epochs'))
    model_type = config.get('model')
    nu_enabled = config.get('nu_enabled')
    plot_save_freq = int(config.get('plot_save_freq'))
    num_plots = 0 # Initialize to zero

    while in_progress:
        # The actual training loop
        old_state = game.board.get_state() # Get the current state
        move = agent.get_move(old_state) # Get the next move
        reward, game_over, score = game.play_step(move) # Play the game step
        if not game_over:
            # Keep playing, this epoch isn't over
            new_state = game.board.get_state() # Get the new state
            agent.train_short_memory(old_state, move, reward, new_state, game_over) # Train short memory
            agent.remember(old_state, move, reward, new_state, game_over) # Remember
        else:
            # Game over
            agent.remember(old_state, move, reward, new_state, game_over) # Remember
            num_games = int(stats.get('game', 'num_games'))
            all_losses = stats.get('loss', 'all')
            # Collect loss data for matplotlib
            cur_loss = stats.get('trainer', 'loss')
            if len(all_losses) == 0:
                stats.append('loss', 'all', cur_loss)
            else:
                total_loss = 0
                for loss in all_losses:
                    total_loss += loss
                total_loss += cur_loss
                stats.append('loss', 'all', total_loss / (len(all_losses) + 1))

            # Check if we've reached the max epochs (if max_)
            if max_epochs > 0 and max_epochs == num_games:
                in_progress = False # Reached max epochs
                log.log("Reached max epochs (" + str(max_epochs) + "), exiting")

            # Check if we've reached the Nu max epochs (if Nu us enabled)
            if nu_enabled and nu_max_epochs > 0 and nu_max_epochs == num_games:
                log.log("Reached Nu max epochs (" + str(nu_max_epochs) + "), disabling Nu")
                config.set('nu_enabled', False)

            # Save the matplotlib figure with the plots 
            if num_games != 0 and (num_games % plot_save_freq == 0):
                num_plots += 1
                plot.save(num_plots)
            if plot_cnn_enable and (num_games % plot_cnn_freq == 0) and model_type == 'cnn':
                cnn_plot.plot(new_state) # Visualize the CNN feature maps

            # Collect score data
            stats.incr('scores', score)

            # Track the scores for each game
            stats.append('scores', 'all', score)

            # Train long memory
            if enable_long_training:
                agent.train_long_memory()

            # New highscore!!! YAY!
            if score > stats.get('game', 'highscore'):
                stats.set('game', 'highscore', score)
                update_highscore_file(config, num_games, score)

            game.reset() # Reset the game

            print_stats(log, stats, agent, config) # Print some stats

            # Update the plots every plot_freq games
            if num_games % config.get('plot_freq') == 0:
                plot.plot() 

            # Save the model with weights
            if num_games % config.get('checkpoint_freq') == 0:
                checkpoint(config, stats, agent) 

            # Recalculate averages over the last show_summary_freq games
            if num_games % config.get('show_summary_freq') == 0:
                update_avg(log, stats, config)
            
            # Save the stats to file every stats_save_freq games
            if num_games % config.get('stats_save_freq') == 0:
                stats.save()

            # Let the agent know we've finished a game
            agent.played_game(score) 

    cleanup(agent, plot)

def print_stats(log, stats, agent, config):
    summary = ''
    summary += 'AISim #' + str(agent.ini.get('sim_num')) + ': Game {:<4}'.format(stats.get('game', 'num_games'))
    summary += ' Score: {:>3}'.format(stats.get('game', 'last_score'))
    summary += ', Time: {:6.2f}s'.format(stats.get('game', 'game_time'))
    summary += ', Highscore: {:>3}'.format(stats.get('game', 'highscore'))
    summary += ', ReplayMem: {:>9}'.format(stats.get('replay', 'mem_size'))

    long_training_msg = stats.get('trainer', 'long_training_msg')
    trainer_steps = stats.get('trainer', 'steps')
    # Number of training steps for the epoch
    if config.get('trainer_stats'):
        summary += ', Trainer steps {:>5}'.format(long_training_msg, trainer_steps)

    # Number of forward(x) calls to the model for this epoch
    if config.get('model_stats'):
        summary += ', Model steps {:>5}'.format(stats.get('model', 'steps'))

    # Epsilon stats
    if config.get('epsilon_enabled'):
        epsilon_value = stats.get('epsilon', 'value')
        if epsilon_value > 0:
            epsilon_value = round(epsilon_value, 2)
            epsilon_injected = stats.get('epsilon', 'injected')
            summary += ', Epsilon value: {:>5}, injected: {:>4}'.format(epsilon_value, epsilon_injected)

    # NuAlgo stats
    if config.get('nu_enabled'):
        agent.nu_algo.update_status()
        summary += ', Nu: {}'.format(stats.get('nu', 'status'))

    # Epoch reward
    if config.get('show_reward'):
        summary += ', Reward: {:>6}'.format(round(stats.get('game', 'game_reward'), 1))

    # Loss
    if config.get('show_loss'):
        summary += ', Loss: {:>6}'.format(round(stats.get('trainer', 'loss'), 2))
    
    summary += ' - {}'.format(stats.get('game', 'lose_reason'))
    log.log(summary)

def update_avg(log, stats, config):
    recent_freq = config.get('show_summary_freq')
    recent_loss = 0
    for loss in stats.get('recent', 'loss'):
        recent_loss += loss
    recent_loss = round(recent_loss / recent_freq, 4)
    stats.append('avg', 'loss', recent_loss)
    recent_score = 0
    for score in stats.get('recent', 'score'):
        recent_score += score
    recent_score = round(recent_score / recent_freq, 2)
    stats.append('avg', 'score', recent_score)
    
def update_highscore_file(config, num_games, score):
    data_dir = config.get('data_dir')
    sim_num = config.get('sim_num')
    sim_data_dir = os.path.join(data_dir, str(sim_num))
    basename = config.get('highscore_file_basename')
    highscore_file = os.path.join(sim_data_dir, str(sim_num) + basename)
    if os.path.isfile(highscore_file):
        # Highscore file exists, update it
        handle = open(highscore_file, mode='a')
        handle.write(str(num_games) + ',' + str(score) + '\n')
        handle.close()
    else:
        # Highscore file does not exist, create it
        handle = open(highscore_file, mode='w')
        handle.write('Game Number,Score\n')
        handle.write(str(num_games) + ',' + str(score) + '\n')
        handle.close()

if __name__ == "__main__":
    train()