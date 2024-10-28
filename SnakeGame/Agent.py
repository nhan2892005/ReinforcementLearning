import torch
import random
import numpy as np
from collections import deque
from snake_ai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import ANN, Trainer
from utils import plot_score

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
NUM_EPISODES = 80
DISCOUNT_RATE = 0.9

Num_states = 11
Num_actions = 3
Num_hidden_layer = 256

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0                        # for randomness
        self.discount_rate = DISCOUNT_RATE
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = ANN(Num_states, Num_hidden_layer, Num_actions)
        self.trainer = Trainer(model=self.model, learning_rate=LEARNING_RATE, gamma=self.discount_rate)

    def get_state(self, game):
        '''
        State: 11 values
        [
        danger straight, danger right, danger left,
        direction left, direction right, direction up, direction down,
        food left, food right, food up, food down
        ]
        '''
        head_snake = game.snake[0]

        # set points
        left_point = Point(head_snake.x - BLOCK_SIZE, head_snake.y)
        right_point = Point(head_snake.x + BLOCK_SIZE, head_snake.y)
        up_point = Point(head_snake.x, head_snake.y - BLOCK_SIZE)
        down_point = Point(head_snake.x, head_snake.y + BLOCK_SIZE)

        # set directions
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT

        state = [
            # Danger straight
            (dir_right and game.is_collision(right_point)) or
            (dir_left and game.is_collision(left_point)) or
            (dir_up and game.is_collision(up_point)) or
            (dir_down and game.is_collision(down_point)),
            # Danger right
            (dir_up and game.is_collision(right_point)) or
            (dir_down and game.is_collision(left_point)) or
            (dir_left and game.is_collision(up_point)) or
            (dir_right and game.is_collision(down_point)),
            # Danger left
            (dir_down and game.is_collision(right_point)) or
            (dir_up and game.is_collision(left_point)) or
            (dir_right and game.is_collision(up_point)) or
            (dir_left and game.is_collision(down_point)),

            # Move direction
            dir_left, dir_right, dir_up, dir_down,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y   # food down
        ]

        return np.array(state, dtype=int)

    def store(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, game_overs = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)
    def get_action(self, state):
        self.epsilon = NUM_EPISODES - self.n_games
        decision = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            idx_action = random.randint(0, 2) # 0: Straight, 1: Right, 2: Left
            decision[idx_action] = 1
        else:
            init_state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(init_state)
            idx_action = torch.argmax(prediction).item()
            decision[idx_action] = 1
        
        return decision
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # current state
        curr_state = agent.get_state(game)

        # get the action
        action = agent.get_action(curr_state)

        # do the action to get the next state
        reward, game_over, score = game.play_step(action)

        # get the next state
        next_state = agent.get_state(game)

        # train the agent
        agent.train_short_memory(curr_state, action, reward, next_state, game_over)
        # store the experience
        agent.store(curr_state, action, reward, next_state, game_over)

        if game_over:
            game.restart()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.cache_current_process()

            print(f'Game {agent.n_games}, Score: {score}, Record: {record}')
            game.save_gif('snake_game.gif')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_score(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()