from SimConfig import SimConfig
from SimLogger import SimLogger
from SimPlot import SimPlot
from SimStats import SimStats
from AIAgent import AIAgent
from AISnakeGame import AISnakeGame
import time

def cleanup(agent, plot):
    agent.cleanup()
    plot.save()
    print('Simulation complete. Results in ' + agent.ini.get('sim_data_dir') + '.')

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
    log.log("AISim initialization:       [OK]")
    stats = SimStats(config, log)
    agent = AIAgent(config, log, stats)
    game = AISnakeGame(config, log, stats)
    plot = SimPlot(config, log, stats)
    game.reset() # Reset the game
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
            if config.get('max_epochs') and config.get('max_epochs') == stats.get('game', 'num_games'):
                in_progress = False # Reached max epochs
                log.log("Reached max epochs (" + str(config.get('max_epochs')) + "), exiting")
            # Track how often a specific score has been reached
            stats.incr('scores', score)
            # Track the scores for each game
            stats.append('scores', 'all', score)
            agent.train_long_memory()
            if score > stats.get('game', 'highscore'):
                # New highscore!!! YAY!
                stats.set('game', 'highscore', score)
            game.reset() # Reset the game
            agent.played_game(score) # Update the agent
            print_stats(log, stats, agent) # Print some stats
            plot.plot() # Plot some stats

    cleanup(agent, plot)

def print_stats(log, stats, agent):
    summary = ''
    summary += 'AISim (v' + str(agent.ini.get('sim_num')) + '): Game {:<4}'.format(stats.get('game', 'num_games'))
    summary += ' Score: {:<2}'.format(stats.get('game', 'last_score'))
    summary += ' Time(s): {:5.2f}'.format(stats.get('game', 'game_time'))
    summary += ' Highscore: {:<3}'.format(stats.get('game', 'highscore'))
    if not stats.get('epsilon', 'depleted'):
        summary += ' Epsilon: {}'.format(stats.get('epsilon', 'status'))
        agent.reset_epsilon_injected()
    summary += ' - {}'.format(stats.get('game', 'lose_reason'))
    log.log(summary)

if __name__ == "__main__":
    train()