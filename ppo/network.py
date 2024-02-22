import torch.nn.functional as F
from torch import nn
import numpy as np
import torch


class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate = 0.1, training = True):
        super(Actor, self).__init__()

        self.input_layer = nn.Linear(in_dim, 512)
        self.hidden_layer1 = nn.Linear(512, 256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, out_dim)

        self.training = training

        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.drop3 = nn.Dropout(p=drop_rate)


    def forward(self, observation: np.ndarray):
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.tensor(observation, dtype=torch.float)
        else:
            observation_tensor = observation

        act1 = F.relu(self.input_layer(observation_tensor))
        act1 = self.drop1(act1) if self.training else act1
        
        act2 = F.relu(self.hidden_layer1(act1))
        act2 = self.drop2(act2) if self.training else act2

        act3 = F.relu(self.hidden_layer2(act2))
        act3 = self.drop3(act3) if self.training else act3
        
        out = self.output_layer(act3)

        out = F.tanh(out)

        return out

class Critic(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate = 0.1, training = True):
        super(Critic, self).__init__()

        self.input_layer = nn.Linear(in_dim, 512)
        self.hidden_layer1 = nn.Linear(512, 256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, out_dim)

        self.training = training

        self.drop1 = nn.Dropout(p=drop_rate)
        self.drop2 = nn.Dropout(p=drop_rate)
        self.drop3 = nn.Dropout(p=drop_rate)


    def forward(self, observation: np.ndarray, hidden = None):
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.tensor(observation, dtype=torch.float)
        else:
            observation_tensor = observation

        act1 = F.relu(self.input_layer(observation_tensor))
        act1 = self.drop1(act1) if self.training else act1
        
        act2 = F.relu(self.hidden_layer1(act1))
        act2 = self.drop2(act2) if self.training else act2

        act3 = F.relu(self.hidden_layer2(act2))
        act3 = self.drop3(act3) if self.training else act3
        
        out = self.output_layer(act3)


        return out