import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import os

class ANN(nn.Module):
    def __init__(self, num_feature_in, num_hidden_layer, num_feature_out):
        super().__init__()
        self.fc1 = nn.Linear(num_feature_in, num_hidden_layer)
        self.fc2 = nn.Linear(num_hidden_layer, num_feature_out)

    def forward(self, x):
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def cache_current_process(self, filename='model.pth'):
        path = './model'
        if not os.path.exists(path):
            os.makedirs(path)

        filename = os.path.join(path, filename)
        torch.save(self.state_dict(), filename)

class Trainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        train_state = torch.tensor(state, dtype=torch.float)
        train_next_state = torch.tensor(next_state, dtype=torch.float)
        train_reward = torch.tensor(reward, dtype=torch.float)
        train_action = torch.tensor(action, dtype=torch.long)

        if len(train_state.shape) == 1:
            train_state = torch.unsqueeze(train_state, 0)
            train_next_state = torch.unsqueeze(train_next_state, 0)
            train_reward = torch.unsqueeze(train_reward, 0)
            train_action = torch.unsqueeze(train_action, 0)
            game_over = (game_over, )

        prediction = self.model(train_state)

        target = prediction.clone()

        for idx in range(len(game_over)):
            Q_new = train_reward[idx]
            if not game_over[idx]:
                Q_new = train_reward[idx] + self.gamma * torch.max(self.model(train_next_state[idx]))

            target[idx][torch.argmax(train_action[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()