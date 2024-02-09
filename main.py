import random
from agent import start

settings_dict = {
    'MAX_MEMORY': 100_000, # Maximum number of experiences in deque
    'BATCH_SIZE': 1000, # Sample size from MAX_MEMORY
    'LR': 0.01, # Learning rate
    'GAMMA': 0.9, # Gamma
    'INPUT_LAYER_SIZE': 11,
    'HIDDEN_LAYER_SIZE': 512,
    'OUTPUT_LAYER_SIZE': 3,
    'RANDOM1': 80, # Random value 1
    'RANDOM2': 200, # Random value 2

    # Read these settings carefully!
    'MODE': "NEW", # Mode can be NEW or VIEW or CONTINUE
    'BLOCK_SIZE': 20, # Size of snake
    'FILE_NAME': "MM100000_BS1000_LR0.001_gamma0.9_HLS512_R180_R2200_GAMES12_ID55274.pth", # File Name, this field is disregard if mode is Continue
    'GAMES': 0, # Games played, leave as 0 if mode is New, set to number after GAMES in file name otherwise.
    'ID': random.randint(1, 100_000)
}

if __name__ == "__main__":
    start(settings_dict)
