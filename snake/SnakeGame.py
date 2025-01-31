"""
SnakeGame.py

This contains the original Snake Game where a user uses the arrow keys
to control the snake.
"""

import pygame
import random
import time
from enum import Enum
from collections import namedtuple

pygame.init()

# RGB colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0,0,0)

# Board Size
BOARD_WIDTH = 400
BOARD_HEIGHT = 400

# Size of the food blocks and snake segments
BLOCK_SIZE = 20

# Speed of the game
SPEED = 4

# The font file used to render the writing on the game screen
FONT = pygame.font.Font('arial.ttf', 25)

# A point on the board
Point = namedtuple('Point', 'x, y')

# The four directions in the game
class Direction(Enum):
  RIGHT = 1
  LEFT = 2
  UP = 3
  DOWN = 4
    
# The Snake Game class
class SnakeGame:
    
  def __init__(self, w=BOARD_WIDTH, h=BOARD_HEIGHT):
    self.w = w
    self.h = h

    # Initialize the display
    self.display = pygame.display.set_mode((self.w, self.h))
    pygame.display.set_caption('Snake Game')
    self.clock = pygame.time.Clock()

    # Simulation metrics
    self.start_time = 0
    self.elapsed_time = 0
    self.sim_start_time = time.time()

    # init game state
    self.direction = Direction.RIGHT
        
    # Start the snake in the middle of the board
    self.head = Point(self.w/2, self.h/2)
    # Give the snake 3 segments at the beginning of the game
    self.snake = [self.head, 
                  Point(self.head.x-BLOCK_SIZE, self.head.y),
                  Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
    
    # Initialize the game score to 0  
    self.score = 0

    # The location of the foot
    self.food = None

    # Place food in a random location
    self._place_food()

  def _pause_game(self):
    is_paused = True
    # Create the pause loop
    while is_paused:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          self._quit_game()
      keys = pygame.key.get_pressed()
      if keys[pygame.K_SPACE]:
        is_paused = False
      if keys[pygame.K_q]:
        self._quit_game()

  def _place_food(self):
    x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
    y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
    self.food = Point(x, y)
    # Make sure that food is not placed inside the snake
    if self.food in self.snake:
      self._place_food()
        
  def play_step(self):
    # 1. Collect user input
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self._quit_game()
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
          self.direction = Direction.LEFT
        elif event.key == pygame.K_RIGHT:
          self.direction = Direction.RIGHT
        elif event.key == pygame.K_UP:
          self.direction = Direction.UP
        elif event.key == pygame.K_DOWN:
          self.direction = Direction.DOWN
        elif event.key == pygame.K_p:
          self._pause_game()
        elif event.key == pygame.K_q:
          self._pause_game()
        
    # 2. move
    self._move(self.direction) # update the head
    self.snake.insert(0, self.head)
        
    # 3. check if game over, wall collision
    game_over = False
    if self._is_wall_collision():
      game_over = True
      return game_over, self.score
    
    if self._is_snake_collision():
      game_over = True
      return game_over, self.score
            
    # 4. place new food or just move
    if self.head == self.food:
      self.score += 1
      self._place_food()
    else:
      self.snake.pop()
        
    # 5. update ui and clock
    self._update_ui()
    self.clock.tick(SPEED)

    # 6. return game over and score
    return game_over, self.score
    
  def _is_wall_collision(self):
    # Hits wall
    if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or \
      self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
      return True
    
  def _is_snake_collision(self):
    # Hits itself
    if self.head in self.snake[1:]:
      return True
        
    return False

  def _quit_game(self):
    pygame.quit()
    quit()

  def _update_ui(self):
    # Paint the background black
    self.display.fill(BLACK)
    
    # Draw the snake
    for pt in self.snake:
      pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
      # TBD What are 2 and 12???
      pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x+1, pt.y+1, BLOCK_SIZE-2, BLOCK_SIZE-2))

    # Draw the food            
    pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

    # Render the score    
    text = FONT.render("Score: " + str(self.score), True, WHITE)
    self.display.blit(text, [0, 0])

    # Redraw the entire board
    # TODO: Use pygame.display.update() and only update the parts of the
    # screen that have been changed. Faster!
    pygame.display.flip()
        
  def _move(self, direction):
    # Move the snake
    x = self.head.x
    y = self.head.y
    if direction == Direction.RIGHT:
      x += BLOCK_SIZE
    elif direction == Direction.LEFT:
      x -= BLOCK_SIZE
    elif direction == Direction.DOWN:
      y += BLOCK_SIZE
    elif direction == Direction.UP:
      y -= BLOCK_SIZE
            
    self.head = Point(x, y)
            

if __name__ == '__main__':
  game = SnakeGame()
    
  # game loop
  while True:
    game_over, score = game.play_step()
        
    if game_over == True:
      break
        
  print('Final Score', score)
  pygame.quit()