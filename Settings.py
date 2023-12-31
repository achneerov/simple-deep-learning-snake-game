# Settings.py
# vars that can be adjusted

import subprocess

SETTINGS_MAX_MEMORY = 100_000  # max number of experiences in deque, experiences are a sequences that look like this (state, action, reward, next_state, done)
SETTINGS_BATCH_SIZE = 1000  # sample of 1000 experiences from MAX_MEMORY, this is done to break correlation, ie if the last few experiences are very similar, used for training long memory
SETTINGS_LR = 0.001  # how slow it learns during training, lower is slower
SETTINGS_gamma = 0.9  # how long term it thinks, on 1.
SETTINGS_HIDDEN_LAYER_SIZE = 256  # size of hidden second layer
SETTINGS_RANDOM1 = 80  # in self.epsilon = SETTINGS_RANDOM1 - self.n_games
SETTINGS_RANDOM2 = 200  # in if random.randint(0, SETTINGS_RANDOM2) < self.epsilon: move = random.randint(0, 2), final_move[move] = 1

MODE = 0  # 0 is to generate a model, 1 is to continue training a model


def file_name_generator():
    filename = f"MM{SETTINGS_MAX_MEMORY}_BS{SETTINGS_BATCH_SIZE}_LR{SETTINGS_LR}_gamma{SETTINGS_gamma}_HLS{SETTINGS_HIDDEN_LAYER_SIZE}_R1{SETTINGS_RANDOM1}_R2{SETTINGS_RANDOM2}.pth"
    return filename


def run_agent():
    try:
        subprocess.run(["/usr/bin/python3", "agent.py"])  # Update this path based on your system's Python location
    except Exception as e:
        print(f"Error running agent.py: {e}")


if __name__ == "__main__":
    run_agent()
