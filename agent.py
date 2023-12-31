import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display
from Settings import (
    SETTINGS_MAX_MEMORY,
    SETTINGS_BATCH_SIZE,
    SETTINGS_LR,
    SETTINGS_gamma,
    SETTINGS_HIDDEN_LAYER_SIZE,
    SETTINGS_RANDOM1,
    SETTINGS_RANDOM2
)

# each game has x moves, each move makes one experience
MAX_MEMORY = SETTINGS_MAX_MEMORY  # max number of experiences in deque, experiences are a sequences that look like this (state, action, reward, next_state, done)
BATCH_SIZE = SETTINGS_BATCH_SIZE  # sample of 1000 experiences from MAX_MEMORY, this is done to break correlation, ie if the last few experiences are very similar
# used for training long memory
LR = SETTINGS_LR  # how slow it learns during training, lower is slower
plt.ion()


# for plotting the results as it runs
def plot(scores, mean_scores):
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

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = SETTINGS_gamma  # how long term it thinks, on 1.
        self.memory = deque(maxlen=MAX_MEMORY)  # how much it can remember
        self.model = Linear_QNet(11, SETTINGS_HIDDEN_LAYER_SIZE,
                                 3)  # first is inputs it takes, second is nodes in hidden layer, third is outputs it gives
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

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

        """
        [0, 0, 0,  # No immediate dangers
        True,     # Moving RIGHT
        False,    # Not moving LEFT
        False,    # Not moving UP
        False,    # Not moving DOWN
        True,     # Food is to the left
        False,    # Food is not to the right
        False,    # Food is not above
        False     # Food is not below
        ]
        """

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):  # determine which action the agent should take
        # starts off random, then relies more on learned as more games are played
        # aka more exploration at the beginning.
        self.epsilon = SETTINGS_RANDOM1 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, SETTINGS_RANDOM2) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # at around 300 games it stops being random at all and only relies on learned policy
            state0 = torch.tensor(state, dtype=torch.float)  # make tensor
            prediction = self.model(state0)  # pass through nn
            move = torch.argmax(prediction).item()  # take best move
            final_move[move] = 1  # make move by setting the 0 of the move direction to 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
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
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
