# agent.py
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer, load_model
import matplotlib.pyplot as plt
from IPython import display
import pickle
import os


def plot(scores, mean_scores):
    """
    Plots the training progress with game scores and mean scores over time.

    Args:
        scores (list): List of individual game scores.
        mean_scores (list): List of mean scores over multiple games.

    Returns:
        None
    """
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
    # Represents the intelligent agent controlling the Snake game.
    def __init__(self, settings, memory, model):
        """
        Initializes the Agent with specified settings, memory, and model.

        Args:
            settings (dict): Dictionary containing various settings.
            memory (deque): Replay memory storing past experiences.
            model (Linear_QNet): Neural network model for Q-learning.
        """
        self.n_games = settings['GAMES']
        self.gamma = settings['GAMMA']
        self.memory = memory
        self.model = model
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

    def get_state(self, game, block_size):
        """
        Generates the current state representation based on the game state.

        Args:
            game (SnakeGameAI): The Snake game instance.
            block_size (int): Size of the game block.

        Returns:
            np.array: Array representing the current state.
        """
        head = game.snake[0]  # point in x,y form of head
        point_l = Point(head.x - block_size, head.y)
        point_r = Point(head.x + block_size, head.y)
        point_u = Point(head.x, head.y - block_size)
        point_d = Point(head.x, head.y + block_size)

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


def load_memory(settings):
    """
    Loads the replay memory from a file or creates a new one if not found.

    Args:
        settings (dict): Dictionary containing various settings.

    Returns:
        deque: Replay memory.
    """
    memory_file_name = settings['FILE_NAME'] + ".pth_memory.txt"
    # Create the full path to the memory file
    memory_file_path = os.path.join('./models', str(settings['ID']), memory_file_name)
    if os.path.exists(memory_file_path):
        # Deserialize the list from the .txt file
        with open(memory_file_path, 'rb') as f:
            memory_list = pickle.load(f)

        # Convert the list back to a deque
        memory = deque(memory_list)
    else:
        print(f"Memory file {memory_file_path} not found. Starting with a fresh memory.")
        memory = deque(maxlen=settings['MAX_MEMORY'])
    return memory


def load_plot(settings):
    """
    Loads the plot data from a file or initializes with default values.

    Args:
        settings (dict): Dictionary containing various settings.

    Returns:
        tuple: Tuple containing plot scores, mean scores, total score, and record.
    """
    # Construct the file name for the plot data based on the FILE_NAME from settings
    plot_file_name = settings['FILE_NAME'] + ".pth_plot.txt"

    # Create the full path to the plot data file
    plot_file_path = os.path.join('./models', str(settings['ID']), plot_file_name)

    if os.path.exists(plot_file_path):
        # Deserialize the plot data dictionary from the .txt file
        with open(plot_file_path, 'rb') as f:
            plot_data = pickle.load(f)

        # Extract individual plot data components
        plot_scores = plot_data['plot_scores']
        plot_mean_scores = plot_data['plot_mean_scores']
        total_score = plot_data['total_score']
        record = plot_data['record']
    else:
        print(f"Plot file {plot_file_path} not found. Starting with default plot data.")
        # Initialize default values or handle as per your requirements
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0

    return plot_scores, plot_mean_scores, total_score, record


def start(settings):
    """
    Initiates the Snake game based on the specified settings and mode.

    Args:
        settings (dict): Dictionary containing various settings.

    Returns:
        None
    """
    mode = settings['MODE']

    if mode == "CONTINUE":
        memory = load_memory(settings)
        plot_scores, plot_mean_scores, total_score, record = load_plot(settings)
        model = load_model(settings)

    elif mode == "VIEW":
        model = load_model(settings)
        memory = deque(maxlen=settings['MAX_MEMORY'])
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0

    else:  # if mode == "NEW":
        memory = deque(maxlen=settings['MAX_MEMORY'])
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        model = Linear_QNet(settings['INPUT_LAYER_SIZE'], settings['HIDDEN_LAYER_SIZE'], settings['OUTPUT_LAYER_SIZE'])

    agent = Agent(settings, memory, model)
    game = SnakeGameAI(settings['BLOCK_SIZE'])
    while True:
        # get old state
        state_old = agent.get_state(game, settings['BLOCK_SIZE'])

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game, settings['BLOCK_SIZE'])

        if mode != "VIEW":
            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1

            if mode != "VIEW":
                agent.train_long_memory()
                memory_usage = (len(agent.memory) / settings['MAX_MEMORY']) * 100

                if score > record:
                    record = score
                    settings['GAMES'] = agent.n_games
                    agent.model.save(settings, agent.memory, plot_scores, plot_mean_scores, total_score,
                                     record)

                print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Memory Usage on 100:', memory_usage, 'Random Move %', agent.epsilon / 200)

            plot_scores.append(score)
            total_score += score
            if mode != 'VIEW':
                mean_score = total_score / agent.n_games
            else:
                mean_score = total_score / (agent.n_games - settings['GAMES'])
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


