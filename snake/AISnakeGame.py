"""
AISnakeGame.py

This contains the Snake Game, modified to have the Trainer class run it
with an AI Agent player. It also uses the StartConfig class to externalize
common parameters.
"""

import pygame
import random
from collections import namedtuple
import numpy as np
import time
import sys, os

lib_dir = os.path.dirname(__file__)
sys.path.append(lib_dir)
from AISnakeGameConfig import AISnakeGameConfig
from SnakeGameElement import Direction
from SnakeGameElement import Point

pygame.init()

# RGB colors
WHITE = (255,255,255)
RED = (200,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (255,0,255)
BLACK = (0,0,0)
GREY = (25,25,25)

# Size of the food blocks and snake segments
BLOCK_SIZE = 20

# Game Caption / Title
GAME_TITLE = 'AI Snake Game '

# The font file used to render the writing on the game screen
FONT = pygame.font.Font('arial.ttf', 25)

class AISnakeGame():
  """
  The AI Snake Game class. This class implements a version of the Snake
  Game that has been modified to have it run by the Trainer class which
  uses the AIAgent class as the player. It also loads game parameters
  from the AISnakeGame.ini file using the StartConfig class.
  """
  def __init__(self, ini, log, plot):
    # Logging object
    self.log = log

    # Get a matplotlib plot object, so we can save the plot when the 
    # game quits
    self.plot = plot

    # This is so that our simulations are repeatable
    random.seed(ini.get('random_seed'))

    # Board size
    self.board_width = ini.get('board_width')
    self.board_height = ini.get('board_height')
    
    # Game speed
    self.game_speed = ini.get('game_speed')

    # For headless mode
    self.headless = ini.get('headless')

    # Initialize the display
    self.screen_width = self.board_width
    self.screen_height = self.board_height
    if not self.headless:
      self.display = pygame.display.set_mode((self.board_width, self.board_height))
      pygame.display.set_caption(GAME_TITLE + ' (v' + str(ini.get('ai_version')) + ')')
    self.clock = pygame.time.Clock()

    # Display periodic status message
    self.print_stats = ini.get('print_stats')
    
    # Simulation metrics
    self.elapsed_time = 0
    self.food = None
    self.game_score = 0
    self.game_moves = 0
    self.lose_reason = 'N/A'
    self.max_moves = ini.get('max_moves')
    self.num_games = 0
    self.num_games_cur = self.num_games
    self.score = 0
    self.sim_start_time = time.time()
    self.sim_score = 0
    self.sim_high_score = 0
    self.sim_time = 0
    self.sim_wall_collision_count = 0
    self.sim_snake_collision_count = 0
    self.sim_exceeded_max_moves_count = 0
    self.start_time = 0
    self.status_iter = ini.get('status_iter')

  def game_speed_decrease(self):
    # PyGame input comes in WAAAAY too fast, throttle it so that
    # this setting doesn't go haywire
    if self.num_games == self.num_games_cur:
      self.game_speed = self.game_speed - 10
      self.log(f"AiSnakeGame: Decreasing game speed to {self.game_speed}")
      self.num_games_cur += 1

  def game_speed_increase(self):
    if self.num_games == self.num_games_cur:
        self.game_speed = self.game_speed + 10
        self.log(f"AiSnakeGame: Increasing game speed to {self.game_speed}")
        self.num_games_cur += 1

  def get_num_games(self):
    """
    Return the current number of games.
    """
    return self.num_games
  
  def get_score(self):
    """
    Return the current game score.
    """
    return self.score

  def is_snake_collision(self, pt=None):
    """
    Check for collisions with the snake.
    """
    if pt is None:
      pt = self.head
    if pt in self.snake[1:]:
      return True
    return False
  
  def is_wall_collision(self, pt=None):
    """
    Check for a wall collision.
    """
    if pt is None:
      pt = self.head
    # hits boundary
    if pt.x > self.board_width - BLOCK_SIZE or pt.x < 0 or \
      pt.y > self.board_height - BLOCK_SIZE or pt.y < 0:

      return True
    return False

  def move(self, action):
    """
    Execute a snake move. 
    action is an enum value [straight, right, left].
    """
    self.direction = self.move_helper(action)
    aPoint = self.move_helper2(self.head.x, self.head.y, self.direction)
    self.head = aPoint

  def move_helper(self, action):
    """
    Based on the action, return the next direction.
    action is an enum value [straight, right, left].
    """
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx = clock_wise.index(self.direction)
    if np.array_equal(action, [1, 0, 0]):
      new_dir = clock_wise[idx] # no change
    elif np.array_equal(action, [0, 1, 0]):
      next_idx = (idx + 1) % 4 # Mod 4 to avoid out of index error
      new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
    else: # [0, 0, 1] ... there are only 3 actions
      next_idx = (idx - 1) % 4 # Again, MOD 4 to avoid out of index error
      new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
    return new_dir

  def move_helper2(self, x, y, direction):
    """
    Based on the x, y, and direction, return the next point of the snake head.
    """
    if direction == Direction.RIGHT:
      x += BLOCK_SIZE
    elif direction == Direction.LEFT:
      x -= BLOCK_SIZE
    elif direction == Direction.DOWN:
      y += BLOCK_SIZE
    elif direction == Direction.UP:
      y -= BLOCK_SIZE
    return Point(x, y)  

  def pause_game(self):
    """
    Pause the game.
    """
    is_paused = True
    # The loop is *FAST* make sure that we only print this info message once
    if self.num_games == self.num_games_cur:
      self.log.log("AISnakeGame: Game paused, press SPACE to continue. Press 'm' to print the models. Press 'q' to quit. Press 'a/z' to speedup/slowdown the game.")
      self.num_games_cur += 1
    # Create pause loop
    while is_paused:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self.quit_game()
      keys = pygame.key.get_pressed()
      if keys[pygame.K_SPACE]:
        is_paused = False
      if keys[pygame.K_q]:
        self.quit_game()
      if keys[pygame.K_s]:
        self.print_status()
      if keys[pygame.K_m]:
        self.print_model()   
      if keys[pygame.K_a]:
        self.game_speed_increase()
      if keys[pygame.K_z]:
        self.game_speed_decrease()


  def place_food(self):
    x = random.randint(0, (self.board_width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
    y = random.randint(0, (self.board_height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
    self.food = Point(x, y)
    if self.food in self.snake:
      self.place_food()

  def play_step(self, action):
    # Track the frame iteration or number of game moves
    self.game_moves += 1

    # 1. collect user input
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self.quit_game()

    keys = pygame.key.get_pressed()
    if keys[pygame.K_p]:
      self.pause_game()
    if keys[pygame.K_q]:
      self.quit_game()
    if keys[pygame.K_s]:
      self.print_status()
    if keys[pygame.K_m]:
      self.print_model()
    if keys[pygame.K_a]:
      self.game_speed_increase
    if keys[pygame.K_z]:
      self.game_speed_decrease

    # 2. move
    self.move(action) # update the head
    self.snake.insert(0, self.head)

    # 3. check if game over, track the reward and the reason the game ended
    reward = 0
    game_over = False
    if self.is_wall_collision():
      game_over = True
      reward = -10
      lose_reason = 'Hit the wall'
      self.lose_reason = lose_reason
      self.agent.ini.set_value('lose_reason', lose_reason)
      self.sim_wall_collision_count += 1
      self.agent.ini.set_value('wall_collision_count', str(self.sim_wall_collision_count))
      return reward, game_over, self.score
    elif self.is_snake_collision():
      game_over = True
      reward = -10
      lose_reason = 'Hit the snake'
      self.lose_reason = lose_reason
      self.agent.ini.set_value('lose_reason', lose_reason)
      self.sim_snake_collision_count += 1
      self.agent.ini.set_value('snake_collision_count', str(self.sim_snake_collision_count))
      return reward, game_over, self.score
    if self.game_moves > self.max_moves*len(self.snake):
      game_over = True
      reward = -10
      lose_reason = 'Excessive moves (' + str(self.max_moves*len(self.snake)) + ')'
      self.lose_reason = lose_reason
      self.agent.ini.set_value('lose_reason', lose_reason)
      self.sim_exceeded_max_moves_count += 1
      self.agent.ini.set_value('exceeded_max_moves_count', str(self.sim_exceeded_max_moves_count))
      return reward, game_over, self.score    

    # 4. place new food or just move
    if self.head == self.food:
      self.score += 1
      reward = 10
      self.place_food()
    else:
      self.snake.pop()

    # 5. update ui and clock
    self.update_ui()
      
    self.clock.tick(self.game_speed)

    # 6. return game over and score
    return reward, game_over, self.score  
        
  def print_model(self):
    """
    Print the neural network model.
    """
    if self.num_games == self.num_games_cur:
      self.agent.l1_model.ascii_print()
      print(self.agent.l1_model)    
      self.agent.l2_model.ascii_print()
      print(self.agent.l2_model)    
      self.num_games_cur += 1

  def print_status(self):
    """
    Print simulation metrics.
    """
    if self.print_stats:
      self.log(f"Total simulation time    : {self.total_sim_time}")
      self.log(f"Total number of games    : {self.num_games}")
      self.log(f"High score               : {self.sim_high_score}")
      self.log(f"Total simulation score   : {self.sim_score}")
      self.log(f"Exceeded max moves count : {self.sim_exceeded_max_moves_count}")
      self.log(f"Wall collision count     : {self.sim_wall_collision_count}")
      self.log(f"Snake collision count    : {self.sim_snake_collision_count}")
      if self.agent.nu_algo.print_stats:
        self.log(f"Nu algorithm score       : {self.agent.nu_algo.get_nu_score()}")
      self.log(f"Average game score       : {self.avg_game_score}")
      self.log(f"Average game time        : {self.avg_game_time} sec")

  def quit_game(self):
    self.sim_score += self.score
    self.sim_time += self.elapsed_time
    self.print_status()
    self.agent.save_checkpoint()
    self.plot.save()
    pygame.quit()
    quit()

  def reset(self):
    """
    (Re)initialize the game state
    """
    # Total score of all games in this simulation run
    self.sim_score += self.score
    self.agent.ini.set_value('total_sim_score', str(self.sim_score))
    # Total simulation time
    sim_min = int(self.sim_time / 60)
    sim_sec = round(self.sim_time % 60, 2)
    self.total_sim_time = str(sim_min) + " min " + str(sim_sec) + " sec"
    self.agent.ini.set_value('total_sim_time', str(self.total_sim_time))
    # Average game score
    if self.num_games == 0:
      self.avg_game_score = 0
    else:
      self.avg_game_score = round(self.sim_score / self.num_games, 2)
    self.agent.ini.set_value('avg_game_score', str(self.avg_game_score))
    # Average game time
    if self.num_games == 0:
      self.avg_game_time = 0
    else:
      self.avg_game_time = round(self.sim_time / self.num_games, 2)
    self.agent.ini.set_value('avg_game_time', str(self.avg_game_time))
    # Current game score
    self.score = 0
    # Number of games
    self.num_games += 1
    self.num_games_cur = self.num_games
    # Simulation runtime (clock time, not CPU time)
    self.sim_time += self.elapsed_time
    self.start_time = time.time()
    
    # Print out some simulation metrics, every status_iter games
    if (self.num_games % self.status_iter) == 0:
      self.print_status()

    # The number of game moves
    self.game_moves = 0

    ## (Re)initialize the game state
    # The direction of the snake
    self.direction = Direction.RIGHT
    self.head = Point(self.board_width/2, self.board_height/2)
    self.snake = [self.head,
      Point(self.head.x-BLOCK_SIZE, self.head.y),
      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
    
    self.place_food()

  def set_agent(self, agent):
    """
    The Trainer uses this to pass in the current instance of the AI agent.
    """
    self.agent = agent

  def update_ui(self):
    self.elapsed_time = round(float((time.time() - self.start_time)), 2)
    if not self.headless:
      # Paint the background black
      self.display.fill(BLACK)
      
      # Draw the snake
      for pt in self.snake:
        pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x+1, pt.y+1, BLOCK_SIZE-2, BLOCK_SIZE-2))

      # Draw the food            
      pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
      pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x+1, self.food.y+1, BLOCK_SIZE-2, BLOCK_SIZE-2))

      # Render the score, number of moves and the game time
      score_string = "Score: " + str(self.score)
      text = FONT.render(score_string + ', Time ' + str(self.elapsed_time) + 's', True, WHITE)
      self.display.blit(text, [0, 0])    
      pygame.display.flip()        
    
    
    



