import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import time

random.seed(1970)

# The runtime environment was setup within venv with the following
# commands:
#
# $ python3 venv ai_dev
# $ . ai_dev/bin/activate
# (ai_dev) $ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# (ai_dev) $ pip3 install pygame, matplotlib, IPython
#
# Run the game with the AI driving:
# (ai_dev) $ python agent.py

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
  RIGHT = 1
  LEFT = 2
  UP = 3
  DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
BOARD_WIDTH = 600
BOARD_HEIGHT = 400

BLOCK_SIZE = 20
SPEED = 100
MAX_ITER = 200 # Used to end the game if the AI goes into a looping pattern
STATUS_ITER = 20 # Print out a simulation status message every STATUS_ITER games

class SnakeGameAI:
    
  def __init__(self, version='N/A', w=BOARD_WIDTH, h=BOARD_HEIGHT):
    self.w = w
    self.h = h
    # init display
    self.display = pygame.display.set_mode((self.w, self.h))
    pygame.display.set_caption('Snake AI (v' + str(version) + ')')
    self.clock = pygame.time.Clock()
    self.start_time = 0
    self.elapsed_time = 0
    self.sim_start_time = time.time()
    self.sim_score = 0
    self.sim_high_score = 0
    self.sim_time = 0
    self.score = 0
    self.num_games = 0
    self.lose_reason = 'N/A'
    self.reset()

  def reset(self):
    # init game state
    self.sim_score += self.score
    self.score = 0
    self.num_games += 1
    self.sim_time += self.elapsed_time
    self.start_time = time.time()
    self.direction = Direction.RIGHT

    # Print out a status message every STATUS_ITER games
    if (self.num_games % STATUS_ITER) == 0:
      self._print_status()
        
    self.head = Point(self.w/2, self.h/2)
    self.snake = [self.head, 
                  Point(self.head.x-BLOCK_SIZE, self.head.y),
                  Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
    
    
    self.food = None
    self._place_food()
    self.frame_iteration = 0

  def _pause_game(self):
    is_paused = True
    # Create pause loop
    while is_paused:
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_SPACE:
            is_paused = False
          if event.key == pygame.K_q:
            self._quit_game()
          if event.key == pygame.K_s:
            self._print_status()
      if event.type == pygame.KEYDOWN:
        if event.type == pygame.QUIT:
          is_paused = False
          self._quit_game()

  def _print_status(self):
    avg_score = int(self.sim_score / self.num_games)
    sim_min = int(self.sim_time / 60)
    sim_sec = self.sim_time % 60
    print(f"Simulation time          : {sim_min} min {sim_sec} sec")
    print(f"Simulation high score    : {self.sim_high_score}")
    print(f"Simulation total score   : {self.sim_score}")
    print(f"Simulation average score : {avg_score}")
    
  def _quit_game(self):
    self.sim_score += self.score
    self.sim_time += self.elapsed_time
    self._print_status()
    pygame.quit()
    quit()

  def _place_food(self):
    x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
    y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
    self.food = Point(x, y)
    if self.food in self.snake:
      self._place_food()
        
  def play_step(self, action):
    # Track the frame iteration
    self.frame_iteration += 1
    # 1. collect user input
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self._quit_game()
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_p:
          self._pause_game()
        elif event.key == pygame.K_q:
          self._quit_game()
        elif event.key == pygame.K_s:
          self._print_status()
        
    # 2. move
    self._move(action) # update the head
    self.snake.insert(0, self.head)
        
    # 3. check if game over and track the reward
    reward = 0
    game_over = False
    if self.is_wall_collision():
      game_over = True
      reward = -10
      self.lose_reason = 'Hit the wall'
      return reward, game_over, self.score
    elif self.is_self_collision():
      game_over = True
      reward = -10
      self.lose_reason = 'Hit the snake'
      return reward, game_over, self.score
    if self.frame_iteration > MAX_ITER*len(self.snake):
      game_over = True
      reward = -10
      self.lose_reason = 'Excessive moves (' + str(MAX_ITER*len(self.snake)) + ')' 
      return reward, game_over, self.score
    
            
    # 4. place new food or just move
    if self.head == self.food:
      self.score += 1
      reward = 10
      self._place_food()
    else:
      self.snake.pop()
        
    # 5. update ui and clock
    self._update_ui()
    self.clock.tick(SPEED)
    
    # 6. return game over and score
    return reward, game_over, self.score
    
  def is_wall_collision(self, pt=None):
    # Handle the danger of the snake hitting the wall
    if pt is None:
      pt = self.head
    # hits boundary
    if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or \
      pt.y > self.h - BLOCK_SIZE or pt.y < 0:
      return True
    return False
    
  def is_self_collision(self, pt=None):
    # Handle the danger of the snake hitting itself
    # hits itself
    if pt is None:
      pt = self.head
    if pt in self.snake[1:]:
      return True
    return False
        
  def _update_ui(self):
    self.display.fill(BLACK)
        
    for pt in self.snake:
      pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
      pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
      pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
      
      score_string = "Score: " + str(self.score)
      self.elapsed_time = int(time.time() - self.start_time)
        
      text = font.render(score_string + ' (' + str(self.elapsed_time) + 's)', True, WHITE)
      self.display.blit(text, [0, 0])
      pygame.display.flip()
        
  def _move(self, action):
    # action is an enum value [straight, right, left]
    
    self.direction = self.move_helper(action)
    aPoint = self.move_helper2(self.head.x, self.head.y, self.direction)
    self.head = aPoint

  def move_helper(self, action):
    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx = clock_wise.index(self.direction)

    if np.array_equal(action, [1, 0, 0]):
      new_dir = clock_wise[idx] # no change
    elif np.array_equal(action, [0, 1, 0]):
      next_idx = (idx + 1) % 4 # MOD 4 to avoid out of index error
      new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
    else: # [0, 0, 1] ... there are only 3 actions
      next_idx = (idx - 1) % 4 # Again, MOD 4 to avoid out of index error
      new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
    return new_dir

  def move_helper2(self, x, y, direction):
    if direction == Direction.RIGHT:
      x += BLOCK_SIZE
    elif direction == Direction.LEFT:
      x -= BLOCK_SIZE
    elif direction == Direction.DOWN:
      y += BLOCK_SIZE
    elif direction == Direction.UP:
      y -= BLOCK_SIZE
    return Point(x, y)  
            

