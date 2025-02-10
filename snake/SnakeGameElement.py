"""
SnakeGameElement.py

"""
from enum import Enum
from collections import namedtuple

class Direction(Enum):
  """
  A simple Enum class that represents a direction in the
  Snake game. It has four values:
  1. RIGHT
  2. LEFT
  3. UP
  4. DOWN
  """
  RIGHT = 1
  LEFT = 2
  UP = 3
  DOWN = 4

"""
A named tuple, Point, with x and y values.
"""
Point = namedtuple("Point", "x y")