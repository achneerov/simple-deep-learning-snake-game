# Snake Game with Q-learning Agent

## Overview
This repository contains a Snake game implementation using Pygame along with an intelligent agent trained using Q-learning to play the game. The agent learns to make decisions on the Snake's movements through reinforcement learning.

## Credits
Credit where credit is due, I followed this tutorial, and have made some modifications and improvements, here is the URL if you are interested: [https://www.youtube.com/watch?v=L8ypSXwyBds](https://www.youtube.com/watch?v=L8ypSXwyBds).

## Files and Structure
- **agent.py**: Contains the Q-learning agent class and functions to interact with the Snake game.
- **game.py**: Implements the SnakeGameAI class, which represents the Snake game environment.
- **model.py**: Defines the Q-network model and QTrainer class for training the agent.
- **main.py**: Entry point to start the Snake game with the Q-learning agent.

## Installation
First, install the required packages from `requirements.txt` using pip:
bash pip3 install -r requirements.txt


## Usage
Go to `main.py`, within it you will see a dictionary called `settings_dict`. For your first use, it is recommended to leave the settings as is.

Run the code. If you are using VSCode, click the run button on the top right. Otherwise, you can run it through the terminal by navigating to the directory and executing:


Two screens will appear:
1. An automatically updating graph showing the performance of the snake.
2. A live view of the snake playing the game. It is suggested to watch it play for  100 rounds, which should take approximately  5 minutes.

After the game has played, you will find new folders with `.pth` model files in the `models` folder. These files represent the trained models.

## Settings Explanation
In the `settings_dict`, there is a setting called `mode` which can be set to `NEW`, `VIEW`, or `CONTINUE`:
- `NEW`: Train a new model (set `FILE_NAME` to an empty string and `GAMES` to  0)
- `VIEW`: View an existing model without updating its training (set `FILE_NAME` to the desired model file name and `GAMES` to the number after `GAMES` in the file name)
- `CONTINUE`: Continue training an existing model (set `FILE_NAME` to the desired model file name and `GAMES` to the number after `GAMES` in the file name)

## Settings Configuration

Depending on the `mode` you choose, the `FILE_NAME` and `GAMES` settings will be configured as follows:

- **If `mode` is set to `NEW`:**
  - Disregard the `FILE_NAME` setting.
  - Set `GAMES` to   0.

- **If `mode` is set to `VIEW` or `CONTINUE`:**
  - Ensure the `FILE_NAME` is the same as a file name in your `models` folder, such as `MM100000_BS1000_LR0.001_gamma0.9_HLS2048_R180_R2200_GAMES1_ID1270.pth`.
  - Do not set the name to any of the files that end in `.txt`.
  - Set `GAMES` to the integer value found after `GAMES` in the file name. For example, if the file name is `MM100000_BS1000_LR0.001_gamma0.9_HLS2048_R180_R2200_GAMES1_ID1270.pth`, set `GAMES` to   1.

Please note that these settings are only applicable when `mode` is set to either `VIEW` or `CONTINUE`. When `mode` is set to `NEW`, the `FILE_NAME` setting is ignored, and `GAMES` should be set to   0.


Do not modify the input and output layer settings unless you are prepared to adjust other parts of the code accordingly.



## Note
The `.txt` files associated with the model files should not be modified or used as `FILE_NAME`. Only the `.pth` files should be used for training modes.
