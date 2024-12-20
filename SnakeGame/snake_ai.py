import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import imageio

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
RED = (200, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.frames = []  # List to hold frames for GIF
        self.restart()

    def restart(self):
        # Init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move
        self._move(action)  # Update the head
        self.snake.insert(0, self.head)
        
        # 3. Check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        self._update_ui()
        
        # Capture the current frame for GIF
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        self.frames.append(np.transpose(frame, (1, 0, 2)))  # Transpose to make it height x width x channels
        
        self.clock.tick(SPEED)
        # 6. Return game over and score
        return reward, game_over, self.score

    def is_collision(self, point=None):
        if point is None:
            point = self.head
        # Hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # Hits itself
        if point in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        curr_idx = directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = directions[curr_idx]  # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (curr_idx + 1) % 4
            new_dir = directions[next_idx]  # Right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (curr_idx - 1) % 4
            new_dir = directions[next_idx]  # Left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        dir_goal = self.direction
        if dir_goal == Direction.RIGHT:
            x += BLOCK_SIZE
        elif dir_goal == Direction.LEFT:
            x -= BLOCK_SIZE
        elif dir_goal == Direction.DOWN:
            y += BLOCK_SIZE
        elif dir_goal == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

    def save_gif(self, filename='snake_game.gif', duration=1):
        # Save the captured frames as a GIF
        imageio.mimsave(filename, self.frames, duration=duration)