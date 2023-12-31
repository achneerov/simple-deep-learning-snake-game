# agent.py

import argparse
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display


def plot(scores, mean_scores):
    plt.ion()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


class Agent:
    def __init__(self, settings):
        self.n_games = settings['GAMES']
        self.epsilon = settings['EPSILON']
        self.gamma = settings['GAMMA']
        self.memory = deque(maxlen=settings['MAX_MEMORY'])
        self.model = Linear_QNet(settings['INPUT_LAYER_SIZE'], settings['HIDDEN_LAYER_SIZE'], settings['OUTPUT_LAYER_SIZE'])
        self.trainer = QTrainer(self.model, lr=settings['LR'], gamma=self.gamma)
        self.max_memory = settings['MAX_MEMORY']
        self.batch_size = settings['BATCH_SIZE']
        self.lr = settings['LR']
        self.gamma = settings['GAMMA']
        self.input_layer_size = settings['INPUT_LAYER_SIZE']
        self.hidden_layer_size = settings['HIDDEN_LAYER_SIZE']
        self.output_layer_size = settings['OUTPUT_LAYER_SIZE']
        self.random1 = settings['RANDOM1']
        self.random2 = settings['RANDOM2']
        self.mode = settings['MODE']

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)  # list of tuples
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):  # determine which action the agent should take
        # starts off random, then relies more on learned as more games are played
        # aka more exploration at the beginning.
        self.epsilon = self.random1 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, self.random2) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # at around 300 games it stops being random at all and only relies on learned policy
            state0 = torch.tensor(state, dtype=torch.float)  # make tensor
            prediction = self.model(state0)  # pass through nn
            move = torch.argmax(prediction).item()  # take best move
            final_move[move] = 1  # make move by setting the 0 of the move direction to 1

        return final_move


def main(settings):
    max_memory = settings['MAX_MEMORY']
    batch_size = settings['BATCH_SIZE']
    lr = settings['LR']
    gamma = settings['GAMMA']
    input_layer_size = settings['INPUT_LAYER_SIZE']
    hidden_layer_size = settings['HIDDEN_LAYER_SIZE']
    output_layer_size = settings['OUTPUT_LAYER_SIZE']
    random1 = settings['RANDOM1']
    random2 = settings['RANDOM2']
    mode = settings['MODE']

    if mode == "NEW":
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent(settings)
        game = SnakeGameAI()
        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save(max_memory, batch_size, lr, gamma, hidden_layer_size, random1, random2)

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agent settings')
    parser.add_argument('--max_memory', type=int, default=100_000, help='Maximum number of experiences in deque')
    parser.add_argument('--batch_size', type=int, default=1000, help='Sample size from MAX_MEMORY')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Gamma value')
    parser.add_argument('--input_layer_size', type=int, default=11, help='Input layer size')
    parser.add_argument('--hidden_layer_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--output_layer_size', type=int, default=3, help='Output layer size')
    parser.add_argument('--random1', type=int, default=80, help='Random value 1')
    parser.add_argument('--random2', type=int, default=200, help='Random value 2')
    parser.add_argument('--mode', type=str, default="NEW", help='Mode value')
    parser.add_argument('--epsilon', type=float, default=0, help='Should be 0')
    parser.add_argument('--games', type=int, default=0, help='Number of games')

    args = parser.parse_args()

    settings_dict = {
        'MAX_MEMORY': args.max_memory,
        'BATCH_SIZE': args.batch_size,
        'LR': args.lr,
        'GAMMA': args.gamma,
        'INPUT_LAYER_SIZE': args.input_layer_size,
        'HIDDEN_LAYER_SIZE': args.hidden_layer_size,
        'OUTPUT_LAYER_SIZE': args.output_layer_size,
        'RANDOM1': args.random1,
        'RANDOM2': args.random2,
        'MODE': args.mode,
        'EPSILON': args.epsilon,
        'GAMES': args.games
    }


    main(settings_dict)
