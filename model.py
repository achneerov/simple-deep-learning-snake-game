# model.py

import torch
import torch.nn as nn  # contains classes to help build neural network
import torch.optim as optim
import torch.nn.functional as F
import os
import pickle
from os.path import join


def file_name_generator(max_memory, batch_size, lr, gamma, hidden_layer_size, random1, random2, games, ID):
    filename = f"MM{max_memory}_BS{batch_size}_LR{lr}_gamma{gamma}_HLS{hidden_layer_size}_R1{random1}_R2{random2}_GAMES{games}_ID{ID}.pth"
    return filename


# model and trainer template

class Linear_QNet(nn.Module):  # base class for neural network modules
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # calls constructor from nn.module
        self.linear1 = nn.Linear(input_size, hidden_size)  # connect first two sets of nodes
        self.linear2 = nn.Linear(hidden_size, output_size)  # connect last two sets of nodes

    def forward(self,
                x):  # basically passes x through linear1, then passes the output into linear2, then returns the final output
        # get the next move
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, settings, memory, plot_scores, plot_mean_scores, total_score, record):
        file_name = file_name_generator(settings['MAX_MEMORY'], settings['BATCH_SIZE'], settings['LR'],
                                        settings['GAMMA'], settings['HIDDEN_LAYER_SIZE'], settings['RANDOM1'],
                                        settings['RANDOM2'], settings['GAMES'], settings['ID'])
        model_folder_path = join('./models', str(settings['ID']))  # Added closing parenthesis
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

        memory_file_name = file_name + "_memory.txt"

        # Convert deque to list for saving
        memory_list = list(memory)

        with open(memory_file_name, 'wb') as f:
            pickle.dump(memory_list, f)

        plot_file_name = file_name + "_plot.txt"
        plot_data = {
            'plot_scores': plot_scores,
            'plot_mean_scores': plot_mean_scores,
            'total_score': total_score,
            'record': record
        }
        with open(plot_file_name, 'wb') as f:
            pickle.dump(plot_data, f)


def load_model(settings):
    file_name = settings['FILE_NAME']
    file_name = file_name + '.pth'
    input_size = settings['INPUT_LAYER_SIZE']
    hidden_size = settings['HIDDEN_LAYER_SIZE']
    output_size = settings['OUTPUT_LAYER_SIZE']

    model = Linear_QNet(input_size, hidden_size, output_size)

    # Define the path to the .pth file
    file_path = os.path.join('./models', str(settings['ID']), file_name)


    # Check if the file exists
    if os.path.exists(file_path):
        # Load the model state dictionary
        model.load_state_dict(torch.load(file_path))
        print(f"Model loaded successfully from {file_path}.")
    else:
        print(f"Error: Model file {file_path} not found.")

    return model


class QTrainer:  # how the model will be trained.
    def __init__(self, model, lr, gamma):
        self.lr = lr  # size of steps taken toward the final solution, smaller means model will take longer to train.
        self.gamma = gamma  # used for balancing immediate rewards and future rewards. higher gamma is more far-sighted, lower gamma is short-sighted
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # streamlines updating parameters during training
        self.criterion = nn.MSELoss()  # set the evaluation metric, we still set what reward and punishment is, but this works on that deeper

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)  # convert inputs to tensors (matrices)
        next_state = torch.tensor(next_state, dtype=torch.float)  # convert inputs to tensors (matrices)
        action = torch.tensor(action,
                              dtype=torch.long)  # convert inputs to tensors (matrices), long because its discrete
        reward = torch.tensor(reward, dtype=torch.float)  # convert inputs to tensors (matrices)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state,
                                    0)  # transforms tensor from 1d to 2d, basically each element becomes its own array with one element
            next_state = torch.unsqueeze(next_state,
                                         0)  # transforms tensor from 1d to 2d, basically each element becomes its own array with one element
            action = torch.unsqueeze(action,
                                     0)  # transforms tensor from 1d to 2d, basically each element becomes its own array with one element
            reward = torch.unsqueeze(reward,
                                     0)  # transforms tensor from 1d to 2d, basically each element becomes its own array with one element
            done = (done,)

        ### this block calcualtes what the best move was
        pred = self.model(state)  # gets predicted actions, state tensor passed through nn to get predicted Q vals.
        # it does this by calling the forward method of linear_qnet class
        target = pred.clone()  # copies Q vals to target, the Q vals will be 3 total as there are three outputs
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:  # if element in done is false, i  think if game is done
                Q_new = reward[i] + self.gamma * torch.max(
                    self.model(next_state[i]))  # update Q by adding reward + gamma*nextBestMove
            target[i][
                torch.argmax(action[i]).item()] = Q_new  # find action with highest Q value, setting it to next move
        ###

        self.optimizer.zero_grad()  # sets gradioents to 0 before computing gradients of next iteration, aka makes a clean slate from move to move
        loss = self.criterion(target, pred)  # compares pred Q vals and target Q vals
        loss.backward()  # calculates gradients (how much each parameter should change)

        self.optimizer.step()  # update parameters based on gradients
