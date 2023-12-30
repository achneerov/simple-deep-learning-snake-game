import torch
import torch.nn as nn  # contains classes to help build neural network
import torch.optim as optim
import torch.nn.functional as F
import os


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

    def save(self, file_name='model.pth'):  # saves model to folder as a state dictionary, full of weights basically
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:  # hown the model will be trained.
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

        self.optimizer.zero_grad() #sets gradioents to 0 before computing gradients of next iteration, aka makes a clean slate from move to move
        loss = self.criterion(target, pred) # compares pred Q vals and target Q vals
        loss.backward() #calculates gradients (how much each parameter should change)

        self.optimizer.step() #update parameters based on gradients
