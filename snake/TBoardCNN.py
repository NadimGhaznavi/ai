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

EMPTY_VALUE = 0.0
FOOD_VALUE = 0.33
SNAKE_VALUE = 0.99

class TBoard():

  def __init__(self, width, height, block_size):
    self.width = int(width) // block_size
    self.height = int(height) // block_size
    self.block_size = block_size

    # Construct the board
    np_board = np.zeros((self.width, self.height), dtype=np.float32)
    self.board = torch.from_numpy(np_board)

    # Attributes for the food
    self.food = Point(0, 0)

    # Attributes for the snake
    self.snake = None
    
  def delete_food(self):
    self.board[self.food.x][self.food.y] = EMPTY_VALUE

  def delete_snake(self):
    for seg in self.snake:
      self.board[seg.x][seg.y] = EMPTY_VALUE

  def get_board(self):
    return self.board 
 
  def display(self):
    pr_str = ''
    for row in self.board:
      pr_str += 'TBoard: '
      for col in row:
        pr_str += f'{col.item():.1f}|'
      #pr_str += '\n' + 50 * '-' + '\n'
      pr_str += '\n'
    print(pr_str)

  def update_snake(self, snake):
    # Remove the old snake
    self.delete_snake()
    # Add the new snake
    for seg in snake:
      self.board[seg.x][seg.y] = SNAKE_VALUE
    self.snake = snake
  
  def update_food(self, food):
    # Remove the old food
    self.delete_food()
    # Add the new food
    x, y = food.x, food.y
    self.board[x][y] = FOOD_VALUE
    self.food = food

if __name__ == '__main__':
  board = TBoard(400, 400, 20)
  #print(f'board.size() returns  =>  {board.size()}')
  t_board = board.get_board()
  print(t_board)
  
  t_board[5][5] = 9.9
  print(t_board)
  board.display()
  #print(t_board)
  #print(t_board[0][20].item())
  
  
  #plt.imshow(board.squeeze(), cmap='gray_r')
  #plt.show()
  
