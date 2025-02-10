"""
TBoardCNN.py

A class that stores the board state in a numpy.ndarray and
as a torch tensor.
"""
import torch
from torch import tensor
import numpy as np
from SnakeGameElement import Point
import matplotlib.pyplot as plt
import time

EMPTY_VALUE = 0.0
FOOD_VALUE = 0.33
SNAKE_VALUE = 0.99
BOMB_VALUE = 0.11

# RGB colors
WHITE = (255,255,255)
RED = (200,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (255,0,255)
BLACK = (0,0,0)
GREY = (25,25,25)

Theme = 'Classic'
Theme = 'Dark'

if Theme == 'Light':
  BLACK = (255,255,255)
  WHITE = (255,255,255)
  BLUE = (255,255,255)
  GREEN = (0,0,0)
elif Theme == 'Dark':
  BLACK = (0,0,0)
  WHITE = (223,223,223)
  BLUE = (0,0,128)
  RED = (200,0,0)
  BOMB = (100,0,0)
  GREEN = (0,128,0)


class TBoard():

  def __init__(self, width, height, block_size):
    self.width = int(width) // block_size
    self.height = int(height) // block_size
    self.block_size = block_size

    # Construct the board
    self.board = torch.from_numpy(np.zeros((self.width, self.height), dtype=np.float32))

    # Attributes for the food
    self.food = None

    # Attributes for the snake
    self.snake = None
    
  def ascii_display(self):
    pr_str = ''
    for row in self.board:
      pr_str += 'TBoard: '
      for col in row:
        pr_str += f'{col.item():.1f}|'
      #pr_str += '\n' + 50 * '-' + '\n'
      pr_str += '\n'
    print(pr_str)

  def delete_food(self, food):
    if food:
      x = food.x // self.block_size
      y = food.y // self.block_size
      self.board[x][y] = EMPTY_VALUE
      self.pygame.draw.rect(self.display, BLACK, self.pygame.Rect(food.x, food.y, self.block_size, self.block_size))
      self.pygame.draw.rect(self.display, BLACK, self.pygame.Rect(food.x+1, food.y+1, self.block_size-2, self.block_size-2))

  def delete_food_bomb(self, food):
    if food:
      for a in range(-1,1):
        for b in range(-1,1):
          x = (food.x + a) * self.block_size
          y = (food.y + b) * self.block_size
          if x >= 0 and x < self.width and y >= 0 and y < self.height:
            self.update_food(Point(x, y), BOMB)
            self.board[a][b] = EMPTY_VALUE

  def delete_snake(self, snake):
    if snake:
      for seg in snake:
        x = int(seg.x) // self.block_size
        y = int(seg.y) // self.block_size
        self.board[x][y] = EMPTY_VALUE
        self.pygame.draw.rect(self.display, BLACK, self.pygame.Rect(seg.x, seg.y, self.block_size, self.block_size))
        self.pygame.draw.rect(self.display, BLACK, self.pygame.Rect(seg.x+1, seg.y+1, self.block_size-2, self.block_size-2))

  def get_state(self):
    # Flatten the 20x20 into a 1D array
    return self.board.reshape(1, -1)

  def reset(self):
    self.display.fill(BLACK)

  def set_display(self, display):
    self.display = display

  def set_pygame(self, pygame):
    self.pygame = pygame

  def update_snake(self, snake):
    # Remove the old snake
    self.delete_snake(snake)
    self.snake = snake
    # Update the snake
    for seg in self.snake:
      x = int(seg.x) // self.block_size
      y = int(seg.y) // self.block_size
      self.board[x][y] = SNAKE_VALUE
      self.pygame.draw.rect(self.display, GREEN, self.pygame.Rect(seg.x, seg.y, self.block_size, self.block_size))
      self.pygame.draw.rect(self.display, BLUE, self.pygame.Rect(seg.x+1, seg.y+1, self.block_size-2, self.block_size-2))
  
  def update_food(self, food, reward=FOOD_VALUE):
    # Remove the old food
    self.delete_food(food)
    # Add the new food
    self.food = food
    x = food.x // self.block_size
    y = food.y // self.block_size
    self.board[x][y] = reward
    # Draw the food on the display
    self.pygame.draw.rect(self.display, GREEN, self.pygame.Rect(food.x, food.y, self.block_size, self.block_size))
    self.pygame.draw.rect(self.display, RED, self.pygame.Rect(food.x+1, food.y+1, self.block_size-2, self.block_size-2))
      

if __name__ == '__main__':
  board = TBoard(400, 400, 20)
  food = Point(200,200)
  board.update_food(food)
  t_board = board.get_board()
  plt.imshow(t_board, cmap='gray_r')
  plt.show()
  time.sleep(2)
  food = Point(300,300)
  board.update_food(food)
  t_board = board.get_board()
  plt.imshow(t_board, cmap='gray_r')
  plt.show()

