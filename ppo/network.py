import torch.nn.functional as F
from torch import nn
import numpy as np
import torch


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.input_layer = nn.Linear(in_dim, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, out_dim)


    def forward(self, observation: np.ndarray):
        print('forward', observation)
        observation_tensor = torch.tensor(observation, dtype=torch.float)

        act1 = F.relu(self.input_layer(observation_tensor))
        act2 = F.relu(self.hidden_layer(act1))
        out = self.output_layer(act2)

        return out