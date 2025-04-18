from time import time
import random
import pygame
from TBoard import TBoard
from GameElements import Direction, Point
import numpy as np

class AISnakeGame():

    def __init__(self, ini, log, stats):
        self.ini = ini
        self.log = log
        self.stats = stats
        self.pygame = pygame
        self.pygame.init()
        self.log.log('AISnakegame initialization: [OK]')
        self.board = TBoard(ini, log, stats, self.pygame)
        self.model = None # Only used to print out it's structure on demand
        self.headless = False # If True, the game will run without updating the display
        self.init_stats()
        self.game_reward = 0
        self.agent = None
        random.seed(ini.get('random_seed'))

    def get_direction(self):
        return self.direction
    
    def get_distance(self, point1, point2):
        return abs(point1.x - point2.x) + abs(point1.y - point2.y)

    def init_stats(self):
        self.stats.set('game', 'score', 0)
        self.stats.set('game', 'num_games', 0)
        self.stats.set('game', 'game_moves', 0)
        self.stats.set('game', 'start_time', time())
        self.stats.set('game', 'highscore', 0)
        self.stats.set('game', 'wall_collision_count', 0)
        self.stats.set('game', 'snake_collision_count', 0)
        self.stats.set('game', 'exceeded_max_moves_count', 0)
        self.stats.set('game', 'lose_reason', '')
                       
    def move(self, action):
        self.stats.incr('game', 'game_moves')
        self.direction = self.move_helper(action)
        self.old_head = self.head
        self.head = self.move_helper2(self.head.x, self.head.y, self.direction)
    
    def move_helper(self, action):
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
        if direction == Direction.RIGHT:
            x += 1
        elif direction == Direction.LEFT:
            x -= 1
        elif direction == Direction.DOWN:
            y += 1
        elif direction == Direction.UP:
            y -= 1
        return Point(x, y)  
        
    def pause_game(self):
        is_paused = True
        print("Game paused...")
        print(" - SPACE bar to resume")
        print(" - q to quit")
        print(" - d to pause the display, but resume the simulation")
        print(" - u to unpause the display, and resume the simulation")
        print(" - m to print the model")
        print(" - r to print replay memory stats")
        print(" - a to increase speed")
        print(" - z to decrease speed")
        while is_paused:
            for event in self.pygame.event.get():
                if event.type == self.pygame.QUIT:
                    self.quit_game()
                if event.type == self.pygame.KEYDOWN:
                    if event.key == self.pygame.K_SPACE:
                        is_paused = False
                        self.log.log("Resuming the simulation...")
                    if event.key == self.pygame.K_d:
                        self.headless = True
                        self.board.set_headless(True)
                        is_paused = False
                        self.log.log("Resuming the simulation...")
                    if event.key == self.pygame.K_u:
                        self.headless = False
                        self.board.set_headless(False)
                        is_paused = False
                        self.log.log("Resuming the simulation...")
                    if event.key == self.pygame.K_q:
                        self.quit_game()
                    if event.key == self.pygame.K_a:
                        self.board.incr_speed()
                    if event.key == self.pygame.K_z:
                        self.board.decr_speed()
                    if event.key == self.pygame.K_m:
                        self.print_model()
                    if event.key == self.pygame.K_r:
                        self.agent.memory_stats()

    def place_food(self):
        x = random.randint(0, self.ini.get('board_width') - 1) 
        y = random.randint(0, self.ini.get('board_height') - 1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def play_step(self, action):
        self.stats.incr('game', 'game_moves')
        # 1. collect user input
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                self.quit_game()
            if event.type == self.pygame.KEYDOWN:
                if event.key == self.pygame.K_p:
                    self.pause_game()

        # 2. move
        self.move(action) # update the head and old_head
        self.snake.insert(0, self.head)
        self.board.update_snake(self.snake, self.direction)
        snake_length = len(self.snake)
        max_moves = self.ini.get('max_moves')

        # 3. check if game over, track the reward and the reason the game ended
        reward = 0
        game_over = False
        board_squares = self.ini.get('board_height') * self.ini.get('board_width')

        ## Check for "game over" states
        if self.board.is_wall_collision():
            # Wall collision
            game_over = True
            reward = self.ini.get('reward_wall_collision')
            lose_reason = f'Hit the wall  ({reward})'
            self.stats.incr('game', 'wall_collision_count')

        elif self.board.is_snake_collision():
            # Snake collision
            game_over = True
            reward = self.ini.get('reward_snake_collision') - (snake_length * self.ini.get('reward_snake_multiplier'))
            max_reward = self.ini.get('reward_snake_collision_max')  # Set an upper limit, otherwise loss gets crazy
            reward = max(reward, max_reward)
            lose_reason = f'Hit the snake ({reward})'
            self.stats.set('game', 'lose_reason', lose_reason)
            self.stats.incr('game', 'snake_collision_count')

        if self.stats.get('game', 'game_moves') > max_moves * snake_length:
            # Exceeded max moves
            game_over = True
            reward = self.ini.get('reward_excessive_move') 
            lose_reason = 'Exceeded max {} moves ({})'.format(max_moves * snake_length, reward)
            self.stats.set('game', 'lose_reason', lose_reason)
            self.stats.incr('game', 'exceeded_max_moves_count')
        
        if game_over == True:
            # Game is over: Snake or wall collision or exceeded max moves
            self.game_reward += reward
            self.stats.set('game', 'lose_reason', lose_reason)
            self.stats.set('game', 'game_reward', round(self.game_reward, 1))
            self.stats.append('recent', 'score', self.stats.get('game', 'score'))
            return reward, game_over, self.stats.get('game', 'score')

        ## Game is not over, lets see what else is going on

        if self.board.is_snake_collision_close():
            # The next move will place the snake head next to the snake body.
            #print(".", flush=True,end='')
            reward += self.ini.get('reward_snake_collision_close')

        if self.head == self.food:
            # We found food!!
            self.stats.incr('game', 'score')
            reward += self.ini.get('reward_food')
            self.place_food()

        else:
            # Small reward for staying alive
            reward += self.ini.get('reward_move') * (snake_length / board_squares)
            self.snake.pop()

        # Update move reward
        old_distance = self.get_distance(self.old_head, self.food)
        new_distance = self.get_distance(self.head, self.food)
        if new_distance < old_distance:
            reward += self.ini.get('reward_move_closer')
        elif new_distance > old_distance:
            reward += self.ini.get('reward_move_further')
        
        self.game_reward = self.game_reward + reward
        self.stats.set('game', 'game_reward', round(self.game_reward, 1))

        
        # 5. update the ui and clock
        self.board.reset()
        self.board.update_snake(self.snake, self.direction)
        self.board.update_food(self.food)
        self.board.update_score(self.stats.get('game', 'score'))
        self.board.refresh()

        # 6. return game over flag, reward and the score for this move
        return reward, game_over, self.stats.get('game', 'score')

    def print_model(self):
        print(self.model)

    def quit_game(self):
        self.stats.set('sim', 'sim_time', time() - self.stats.get('sim', 'start_time'))
        self.board.quit_game()

    def reset(self):
        self.board.reset()
        self.game_reward = 0
        self.stats.set('game', 'game_time', round(time() - self.stats.get('game', 'start_time'), 2))
        self.stats.set('game', 'last_score', self.stats.get('game', 'score'))
        self.stats.set('game', 'score', 0)
        self.stats.set('game', 'game_moves', 0)
        self.stats.set('game', 'start_time', time())
        self.stats.set('sim', 'start_time', time())
        self.stats.incr('game', 'num_games')
        self.direction = Direction.RIGHT
        self.head = Point(self.board.width/2, self.board.height/2)
        self.snake = [self.head, Point(self.head.x - 1, self.head.y), Point(self.head.x - 2, self.head.y)]
        self.board.update_snake(self.snake, self.direction)
        self.place_food()
        self.board.update_food(self.food)
        self.board.refresh()

    def set_agent(self, agent):
        self.agent = agent

    def set_model(self, model):
        self.model = model


