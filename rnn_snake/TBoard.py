"""
TBoard.py
"""
import torch
import numpy as np
import pygame

# Colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)
YELLOW = (255,0,255)
BLACK = (0,0,0)
GREY = (25,25,25)

# Representation of the board state
EMPTY_VALUE = 0.0
FOOD_VALUE = 0.33
SNAKE_VALUE = 0.99

class TBoard():
    def __init__(self, ini, log, stats, pygame):
        self.ini = ini
        self.log = log
        self.stats = stats
        self.pygame = pygame
        # Setup the board
        self.block_size = ini.get('board_square') # Size of each "square" on the board
        self.width = ini.get('board_width') # Board width in squares
        self.height = ini.get('board_height') # Board height in squares
        self.board = torch.from_numpy(np.zeros((self.width, self.height), dtype=np.float32))
        self.food = None # Initialize food
        self.snake = None # Initialize snake
        # Setup pygme
        self.clock = None # Pygame clock
        self.init_pygame()
        self.log.log('TBoard initialization:      [OK]')

    def delete_food(self, food):
        if food:
            x = int(food.x)
            y = int(food.y)
            block_size = self.block_size
            self.board[x][y] = EMPTY_VALUE
            self.pygame.draw.rect(self.display, BLACK, [x * block_size, y * block_size, block_size, block_size])

    def delete_snake(self, snake):
        if snake:
            block_size = self.block_size
            for seg in snake:
                x = int(seg.x)
                y = int(seg.y)
                if x >= 0 and x < self.width and y >= 0 and y < self.height:
                    self.board[x][y] = EMPTY_VALUE
                    self.pygame.draw.rect(self.display, BLACK, [x * block_size, y * block_size, block_size, block_size])

    def get_state(self):
        return self.board.reshape(1, -1)[0]

    def init_pygame(self):
        # Setup the clock
        self.clock = self.pygame.time.Clock()
        # Font used for rendering text
        self.font = self.pygame.font.Font('arial.ttf', 25)
        # The pygame display screen
        self.display = self.pygame.display.set_mode((self.width * self.block_size, self.height * self.block_size))
        # Set the title of the window
        self.pygame.display.set_caption(self.ini.get('game_title'))
        
    def is_snake_collision(self):
        if self.head in self.snake[1:]:
            self.stats.incr('game', 'snake_collision_count')
            return True
        return False

    def is_wall_collision(self):
        if self.head.x > self.width or self.head.x < 0 or \
            self.head.y > self.height or self.head.y < 0:
            self.stats.incr('game', 'wall_collision_count')
            return True
        return False

    def quit_game(self):
        self.pygame.quit()
        quit()

    def refresh(self):
        self.clock.tick(self.ini.get('game_speed'))
        self.pygame.display.flip()
        self.pygame.display.update()

    def update_score(self, score):
        text = self.font.render("Score: " + str(score - 1), True, BLACK)
        self.display.blit(text, [0, 0])
        text = self.font.render("Score: " + str(score), True, WHITE)
        self.display.blit(text, [0, 0])

    def update_food(self, food):
        # Remove the old food
        self.delete_food(food)
        # Add the new food
        x = int(food.x)
        y = int(food.y)
        block_size = self.block_size
        self.board[x][y] = FOOD_VALUE
        self.pygame.draw.rect(self.display, GREEN, [x * block_size, y * block_size, block_size, block_size])
        self.pygame.draw.rect(self.display, RED, [(x * block_size) + 2, (y * block_size) + 2, block_size - 4, block_size - 4])
    
    
    def update_snake(self, snake):
        # Remove the old snake
        self.delete_snake(snake)
        # Add the new snake
        self.head = snake[0]
        self.snake = snake
        block_size = self.block_size
        for seg in snake:
            x = int(seg.x)
            y = int(seg.y)
            if x >= 0 and x < self.width and y >= 0 and y < self.height:
                self.board[x][y] = SNAKE_VALUE
                self.pygame.draw.rect(self.display, GREEN, [x * block_size, y * block_size, block_size, block_size])
                self.pygame.draw.rect(self.display, BLUE, [(x * block_size) + 2, (y * block_size) + 2, block_size - 4, block_size - 4])

    def reset(self):
        self.display.fill(BLACK)
