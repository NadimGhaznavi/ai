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
GAME_TITLE = 'AI Snake Game'

# The font file used to render the writing on the game screen
FONT = pygame.font.Font('arial.ttf', 25)

class AISnakeGame():
  """
  The AI Snake Game class. This class implements a version of the Snake
  Game that has been modified to have it run by the Trainer class which
  uses the AIAgent class as the player. It also loads game parameters
  from the AISnakeGame.ini file using the StartConfig class.
  """
  def __init__(self, ai_version):
    # Get AISnakeGame.ini file settings
    ini = AISnakeGameConfig()
        
    # This is so that our simulations are repeatable
    random.seed(ini.get('random_seed'))

    # Board size
    self.board_width = ini.get('board_width')
    self.board_height = ini.get('board_height')
    
    # Game speed
    self.game_speed = ini.get('game_speed')

    # Initialize the display
    self.screen_width = self.board_width
    self.screen_height = self.board_height
    self.display = pygame.display.set_mode((self.board_width, self.board_height))
    pygame.display.set_caption(GAME_TITLE + ' (v' + str(ai_version) + ')')
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
    self.score = 0
    self.sim_start_time = time.time()
    self.sim_score = 0
    self.sim_high_score = 0
    self.sim_time = 0
    self.sim_wall_collision_count = 0
    self.sim_snake_collision_count = 0
    self.sim_exceeded_max_moves_count = 0
    self.start_time = 0
    self.sim_save_checkpoint_freq = ini.get('sim_save_checkpoint_freq')
    self.status_iter = ini.get('status_iter')

  def game_speed_decrease(self):
    self.game_speed = self.game_speed - 10

  def game_speed_increase(self):
    self.game_speed = self.game_speed + 10

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
      self.game_speed_increase()
    if keys[pygame.K_z]:
      self.game_speed_decrease()

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
      self.agent.set_config('lose_reason', lose_reason)
      self.sim_wall_collision_count += 1
      self.agent.set_config('wall_collision_count', str(self.sim_wall_collision_count))
      return reward, game_over, self.score
    elif self.is_snake_collision():
      game_over = True
      reward = -10
      lose_reason = 'Hit the snake'
      self.lose_reason = lose_reason
      self.agent.set_config('lose_reason', lose_reason)
      self.sim_snake_collision_count += 1
      self.agent.set_config('snake_collision_count', str(self.sim_snake_collision_count))
      return reward, game_over, self.score
    if self.game_moves > self.max_moves*len(self.snake):
      game_over = True
      reward = -10
      lose_reason = 'Excessive moves (' + str(self.max_moves*len(self.snake)) + ')'
      self.lose_reason = lose_reason
      self.agent.set_config('lose_reason', lose_reason)
      self.sim_exceeded_max_moves_count += 1
      self.agent.set_config('exceeded_max_moves_count', str(self.sim_exceeded_max_moves_count))
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
    print(self.agent.model)    

  def print_status(self):
    """
    Print simulation metrics.
    """
    if self.print_stats:
      print(f"Total simulation time    : {self.total_sim_time}")
      print(f"Total number of games    : {self.num_games}")
      print(f"High score               : {self.sim_high_score}")
      print(f"Total simulation score   : {self.sim_score}")
      print(f"Exceeded max moves count : {self.sim_exceeded_max_moves_count}")
      print(f"Wall collision count     : {self.sim_wall_collision_count}")
      print(f"Snake collision count    : {self.sim_snake_collision_count}")
      #print(f"Nu algorithm score       : {self.agent.nu_algo.get_nu_score()}")
      print(f"Average game score       : {self.avg_game_score}")
      print(f"Average game time        : {self.avg_game_time} sec")

  def quit_game(self):
    self.sim_score += self.score
    self.sim_time += self.elapsed_time
    self.print_status()
    self.agent.save_checkpoint()
    pygame.quit()
    quit()

  def reset(self):
    """
    (Re)initialize the game state
    """
    # Total score of all games in this simulation run
    self.sim_score += self.score
    self.agent.set_config('total_sim_score', str(self.sim_score))
    # Total simulation time
    sim_min = int(self.sim_time / 60)
    sim_sec = round(self.sim_time % 60, 2)
    self.total_sim_time = str(sim_min) + " min " + str(sim_sec) + " sec"
    self.agent.set_config('total_sim_time', str(self.total_sim_time))
    # Average game score
    if self.num_games == 0:
      self.avg_game_score = 0
    else:
      self.avg_game_score = round(self.sim_score / self.num_games, 2)
    self.agent.set_config('avg_game_score', str(self.avg_game_score))
    # Average game time
    if self.num_games == 0:
      self.avg_game_time = 0
    else:
      self.avg_game_time = round(self.sim_time / self.num_games, 2)
    self.agent.set_config('avg_game_time', str(self.avg_game_time))
    # Current game score
    self.score = 0
    # Number of games
    self.num_games += 1
    # Simulation runtime (clock time, not CPU time)
    self.sim_time += self.elapsed_time
    self.start_time = time.time()
    
    # Print out some simulation metrics, every status_iter games
    if (self.num_games % self.status_iter) == 0:
      self.print_status()

    # The number of game moves
    self.game_moves = 0

    # Save the model and it's state every sim_save_checkpoint_freq games
    if(self.num_games % self.sim_save_checkpoint_freq) == 0:
      self.agent.save_checkpoint()

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
    self.elapsed_time = round(float((time.time() - self.start_time)), 2)
    text = FONT.render(score_string + ', Moves ' + str(self.game_moves) + \
                         ', Time ' + str(self.elapsed_time) + 's', True, WHITE)
    self.display.blit(text, [0, 0])    
    pygame.display.flip()        
    
    
    



