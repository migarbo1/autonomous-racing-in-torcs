import torch.nn.functional as F
from torch import nn
import numpy as np
import torch


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate = 0.1, test = False):
        super(Actor, self).__init__()

        self.input_layer = nn.Linear(in_dim, 512)
        self.hidden_layer1 = nn.Linear(512, 256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, out_dim)

        self.test = test

        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.drop3 = nn.Dropout(p=drop_rate)


    def forward(self, observation: np.ndarray):
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.tensor(observation, dtype=torch.float)
        else:
            observation_tensor = observation

        act1 = F.relu(self.input_layer(observation_tensor))
        act1 = self.drop1(act1) if not self.test else act1
        
        act2 = F.relu(self.hidden_layer1(act1))
        act2 = self.drop2(act2) if not self.test else act2

        act3 = F.relu(self.hidden_layer2(act2))
        act3 = self.drop3(act3) if not self.test else act3
        
        out = self.output_layer(act3)

        out = F.tanh(out)

        return out

class Critic(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate = 0.1, test = True):
        super(Critic, self).__init__()

        self.input_layer = nn.Linear(in_dim, 512)
        self.hidden_layer1 = nn.Linear(512, 256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, out_dim)

        self.test = test

        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.drop3 = nn.Dropout(p=drop_rate)


    def forward(self, observation: np.ndarray, hidden = None):
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.tensor(observation, dtype=torch.float)
        else:
            observation_tensor = observation

        act1 = F.relu(self.input_layer(observation_tensor))
        act1 = self.drop1(act1) if not self.test else act1
        
        act2 = F.relu(self.hidden_layer1(act1))
        act2 = self.drop2(act2) if not self.test else act2

        act3 = F.relu(self.hidden_layer2(act2))
        act3 = self.drop3(act3) if not self.test else act3
        
        out = self.output_layer(act3)


        return out

class ActorLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate = 0.1, test = False):
        super(ActorLSTM, self).__init__()

        self.input_layer = nn.Linear(in_dim, 256)
        self.hidden_layer1 = nn.Linear(256, 128)
        self.lstm = nn.LSTM(128, 64, 1, batch_first=True)
        self.output_layer = nn.Linear(64, out_dim)

        self.test = test

        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.drop3 = nn.Dropout(p=drop_rate)


    def forward(self, observation: np.ndarray, hp=None, cp=None):
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.tensor(observation, dtype=torch.float)
        else:
            observation_tensor = observation
        
        # adapt single observation to 2D LSTM format
        single_obs = len(observation.shape) == 1

        act1 = F.relu(self.input_layer(observation_tensor))
        act1 = self.drop1(act1) if not self.test else act1
        
        act2 = F.relu(self.hidden_layer1(act1))
        act2 = self.drop2(act2) if not self.test else act2
        
        act3 = torch.unsqueeze(act2, 0) if single_obs else act2

        if hp is None or cp is None:
            lstm_out, (h, c) = self.lstm(act3)
        else:
            lstm_out, (h, c) = self.lstm(act3, (hp, cp))
        
        lstm_out = torch.squeeze(lstm_out, 0) if single_obs else lstm_out

        out = self.output_layer(lstm_out)

        out = F.tanh(out)

        return out, h, c

class CriticLSTM(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate = 0.1, test = True):
        super(CriticLSTM, self).__init__()

        self.input_layer = nn.Linear(in_dim, 256)
        self.hidden_layer1 = nn.Linear(256, 128)
        self.lstm = nn.LSTM(128, 64, 1, batch_first=True)
        self.output_layer = nn.Linear(64, out_dim)

        self.test = test

        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)


    def forward(self, observation: np.ndarray, hp=None, cp=None):
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.tensor(observation, dtype=torch.float)
        else:
            observation_tensor = observation

        # adapt single observation to 2D LSTM format
        single_obs = len(observation.shape) == 1

        act1 = F.relu(self.input_layer(observation_tensor))
        act1 = self.drop1(act1) if not self.test else act1
        
        act2 = F.relu(self.hidden_layer1(act1))
        act2 = self.drop2(act2) if not self.test else act2
        
        act3 = torch.unsqueeze(act2, 0) if single_obs else act2

        if hp is None or cp is None:
            lstm_out, (h, c) = self.lstm(act3)
        else:
            lstm_out, (h, c) = self.lstm(act3, (hp, cp))

        lstm_out = torch.squeeze(lstm_out, 0) if single_obs else lstm_out

        out = self.output_layer(lstm_out)

        return out, h, c