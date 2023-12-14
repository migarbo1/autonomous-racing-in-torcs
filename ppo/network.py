import torch.nn.functional as F
from torch import nn
import numpy as np
import torch


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.input_layer = nn.Linear(in_dim, 256)
        self.hidden_layer1 = nn.Linear(256, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, out_dim)

        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.2)
        self.do3 = nn.Dropout(0.1)


    def forward(self, observation: np.ndarray):
        if isinstance(observation, np.ndarray):
            observation_tensor = torch.tensor(observation, dtype=torch.float)
        else:
            observation_tensor = observation

        act1 = F.relu(self.input_layer(observation_tensor))
        # act1 = self.do1(act1)
        act2 = F.relu(self.hidden_layer1(act1))
        # act2 = self.do2(act2)
        act3 = F.relu(self.hidden_layer2(act2))
        # act3 = self.do3(act3)
        out = self.output_layer(act3)

        # for actions apply activation function
        if out.size(dim=0) == 3:
            sigm_out = out[0:2]
            tanh_out = out[2:]
            tuple_of_activated_parts = (
                F.sigmoid(sigm_out),
                F.tanh(tanh_out)
            )
            out = torch.cat(tuple_of_activated_parts, dim=0)

        return out