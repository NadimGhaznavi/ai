"""
TBoard.py
"""
import torch
import numpy as np
import pygame
from GameElements import Direction, Point

# Colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE = (0,0,200)
GREEN = (0,200,0)
YELLOW = (255,0,255)
BLACK = (0,0,0)
GREY = (25,25,25)

# Representation of the board state
EMPTY_VALUE = 0.0
FOOD_VALUE = 0.4
SNAKE_VALUE = 0.2
SNAKE_HEAD_VALUE = 0.3

class TBoard():
    def __init__(self, ini, log, stats, pygame):
        torch.manual_seed(ini.get('random_seed'))
        self.ini = ini
        self.log = log
        self.stats = stats
        self.pygame = pygame
        self.plot = None
        self.headless = False # If True, the game will run without updating the display
        # Setup the board
        self.block_size = ini.get('board_square') # Size of each "square" on the board
        self.width = ini.get('board_width') # Board width in squares
        self.height = ini.get('board_height') # Board height in squares
        self.board = None
        self.reset_board()
        self.food = Point(3,3) # Initialize food to a random location
        self.snake = None # Initialize snake
        self.head = Point(self.width/2, self.height/2) # Initialize head
        self.direction = Direction.RIGHT
        self.last_dirs = [ 0, 0, 1, 0 ]
        # Setup pygme
        self.clock = None # Pygame clock
        self.init_pygame()
        self.log.log('TBoard initialization:      [OK]')

    def decr_speed(self):
        speed = self.ini.get('game_speed')
        print("Decreasing game speed to " + str(speed - 10))
        self.ini.set('game_speed',speed - 10)

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

    def get_binary(self, bits_needed, some_int):
        # This is used in the state map, the get_state() function.
        some_int = int(some_int)
        bin_str = format(some_int, 'b')
        out_list = []
        for bit in range(len(bin_str)):
            out_list.append(bin_str[bit])
        for zero in range(bits_needed - len(out_list)):
            out_list.insert(0, '0')
        for x in range(bits_needed):
            out_list[x] = int(out_list[x])
        return out_list

    def get_state(self):
        DEBUG = True
        if DEBUG:
            # Snake collision straight ahead
            if (dir_r and self.is_snake_collision(point_r)) or \
                (dir_l and self.is_snake_collision(point_l)) or \
                (dir_u and self.is_snake_collision(point_u)) or \
                (dir_d and self.is_snake_collision(point_d)):
                self.log.log("Snake collision ahead")

            # Snake collision to the right
            if (dir_u and self.is_snake_collision(point_r)) or \
                (dir_d and self.is_snake_collision(point_l)) or \
                (dir_l and self.is_snake_collision(point_u)) or \
                (dir_r and self.is_snake_collision(point_d)):
                self.log.log("Snake collision to the right")

            # Snake collision to the left
            if (dir_d and self.is_snake_collision(point_r)) or \
                (dir_u and self.is_snake_collision(point_l)) or \
                (dir_r and self.is_snake_collision(point_u)) or \
                (dir_l and self.is_snake_collision(point_d)):
                self.log.log("Snake collision to the left")
            
        model_type = self.ini.get('model')
        if model_type == 'rnn' or model_type == 'linear':
            self.plot.set_image_1(self.board)
        elif model_type == 'cnn' or model_type == 'cnnr' or model_type == 'cnnr3' or model_type == 'cnnr4':
            # Return a 3 channel representation of the game state for the CNN.
            # Where the snake body, head and food each get their own channel.
            state = np.zeros((3, self.height, self.width), dtype=np.float32)
            # Snake head, channel 0
            if self.snake is not None and len(self.snake) > 0:
                head = self.head
                state[0, int(head.y), int(head.x)] = 1.0
            # Snake body, channel 1
            if self.snake is not None and len(self.snake) > 1:
                for seg in self.snake[1:]:
                    state[1, int(seg.y), int(seg.x)] = 1.0
            # Food, channel 2
            if self.food is not None:
                state[2, int(self.food.y), int(self.food.x)] = 1.0
            self.plot.set_image_1(self.board)
            return state

        head = self.head
        direction = self.direction
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN
        slb = self.get_binary(7, len(self.snake))
        headxb = self.get_binary(5, head.x)
        headyb = self.get_binary(5, head.y)

        state = [
            # Snake length in binary using 7 bits
            slb[0], slb[1], slb[2], slb[3], slb[4], slb[5], slb[6],
            # Last move direction
            dir_l, dir_r, dir_u, dir_d,

            # Wall collision straight ahead
            (dir_r and self.is_wall_collision(point_r)) or
            (dir_l and self.is_wall_collision(point_l)) or
            (dir_u and self.is_wall_collision(point_u)) or
            (dir_d and self.is_wall_collision(point_d)),

            # Wall collision to the right
            (dir_u and self.is_wall_collision(point_r)) or
            (dir_d and self.is_wall_collision(point_l)) or
            (dir_l and self.is_wall_collision(point_u)) or
            (dir_r and self.is_wall_collision(point_d)),

            # Wall collision to the left
            (dir_d and self.is_wall_collision(point_r)) or
            (dir_u and self.is_wall_collision(point_l)) or
            (dir_r and self.is_wall_collision(point_u)) or
            (dir_l and self.is_wall_collision(point_d)),

            # Snake collision straight ahead
            (dir_r and self.is_snake_collision(point_r)) or
            (dir_l and self.is_snake_collision(point_l)) or
            (dir_u and self.is_snake_collision(point_u)) or
            (dir_d and self.is_snake_collision(point_d)),

            # Snake collision to the right
            (dir_u and self.is_snake_collision(point_r)) or
            (dir_d and self.is_snake_collision(point_l)) or
            (dir_l and self.is_snake_collision(point_u)) or
            (dir_r and self.is_snake_collision(point_d)),

            # Snake collision to the left
            (dir_d and self.is_snake_collision(point_r)) or
            (dir_u and self.is_snake_collision(point_l)) or
            (dir_r and self.is_snake_collision(point_u)) or
            (dir_l and self.is_snake_collision(point_d)),

            # Food location
            self.food.x < self.head.x, # Food left
            self.food.x > self.head.x, # Food right
            self.food.y < self.head.y, # Food up
            self.food.y > self.head.y, # Food down
            self.food.x == self.head.x,
            self.food.x == self.head.x and self.food.y > self.head.y, # Food ahead
            self.food.x == self.head.x and self.food.y < self.head.y, # Food behind
            self.food.y == self.head.y,
            self.food.y == self.head.y and self.food.x > self.head.x, # Food above
            self.food.y == self.head.y and self.food.x < self.head.x, # Food below
            
        ]
        # Previous direction of the snake
        for aDir in self.last_dirs:
            state.append(int(aDir))
        self.last_dirs = [ dir_l, dir_r, dir_u, dir_d ]

        # Head location in binary using 4 bits (x,y)
        for aBit in headxb:
            state.append(int(aBit))
        for aBit in headyb:
            state.append(int(aBit))

        #return torch.from_numpy(np.array(state, dtype=np.float32))
        return np.array(state, dtype='int8')

    def incr_speed(self):
        speed = self.ini.get('game_speed')
        print("Increasing game speed to " + str(speed + 10))
        self.ini.set('game_speed',speed + 10)

    def init_pygame(self):
        # Setup the clock
        self.clock = self.pygame.time.Clock()
        # Font used for rendering text
        self.font = self.pygame.font.Font('arial.ttf', 25)
        # The pygame display screen
        self.display = self.pygame.display.set_mode((self.width * self.block_size, self.height * self.block_size))
        # Set the title of the window
        self.pygame.display.set_caption(self.ini.get('game_title') + ' (v' + str(self.ini.get('sim_num')) + ')')
        
    def is_snake_collision(self, pt=None):
        if not pt:
            pt = self.head
        if pt in self.snake[1:]:
            return True
        return False
    
    def is_snake_collision_close(self, pt=None):
        # Return true if the head is next to the body.
        if pt is None:
            head = self.head
        else:
            head = pt
        if len(self.snake) >= 5:
            body_positions = {(seg.x, seg.y) for seg in self.snake[4:]}  # Use a set for O(1) lookup
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                if (head.x + dx, head.y + dy) in body_positions:
                    return True
        return False

    def is_wall_collision(self, pt=None):
        if not pt:
            x = self.head.x
            y = self.head.y
        else:
            x = pt.x
            y = pt.y
        if x >= self.width or x < 0 or \
            y >= self.height or y < 0:
            return True
        return False

    def quit_game(self):
        self.pygame.quit()
        quit()

    def refresh(self):
        self.clock.tick(self.ini.get('game_speed'))
        if not self.headless:
            self.pygame.display.flip()
            self.pygame.display.update()

    def reset(self):
        self.reset_board()
        if not self.headless:
            self.display.fill(BLACK)

    def reset_board(self):
        self.board = torch.from_numpy(np.zeros((self.width, self.height), dtype=np.float32))

    def set_headless(self, flag):
        self.headless = flag

    def set_plot(self, plot):
        self.plot = plot

    def update_score(self, score):
        if not self.headless:
            text = self.font.render("Score: " + str(score - 1), True, BLACK)
            self.display.blit(text, [0, 0])
            text = self.font.render("Score: " + str(score), True, WHITE)
            self.display.blit(text, [0, 0])

    def update_food(self, food):
        # Remove the old food
        self.delete_food(food)
        # Add the new food
        self.food = food
        x = int(food.x)
        y = int(food.y)
        block_size = self.block_size
        self.board[x][y] = FOOD_VALUE
        self.pygame.draw.rect(self.display, GREEN, [x * block_size, y * block_size, block_size, block_size])
        self.pygame.draw.rect(self.display, RED, [(x * block_size) + 2, (y * block_size) + 2, block_size - 4, block_size - 4])
    
    def update_snake(self, snake, direction):
        self.direction = direction
        # Remove the old snake
        self.delete_snake(snake)
        # Add the new snake
        self.head = snake[0]
        self.snake = snake
        block_size = self.block_size
        head_seg = True
        for seg in snake:
            x = int(seg.x)
            y = int(seg.y)
            if x >= 0 and x < self.width and y >= 0 and y < self.height:
                if head_seg:
                    self.board[x][y] = SNAKE_HEAD_VALUE
                    head_seg = False
                else:
                    self.board[x][y] = SNAKE_VALUE
                self.pygame.draw.rect(self.display, GREEN, [x * block_size, y * block_size, block_size, block_size])
                self.pygame.draw.rect(self.display, BLUE, [(x * block_size) + 2, (y * block_size) + 2, block_size - 4, block_size - 4])

