"""
SnakeGameDirection.py

This file contains a simple Enum class that represents a direction in the
Snake game. It has four values:
1. RIGHT
2. LEFT
3. UP
4. DOWN
"""
from enum import Enum
from collections import namedtuple

class Direction(Enum):
  RIGHT = 1
  LEFT = 2
  UP = 3
  DOWN = 4

Point = namedtuple("Point", "x y")