import torch
import torch.nn.functional as functional
import torch.nn as nn


class NeutralNetwork(nn.Module):
    def __init__(self, n):
        super(NeutralNetwork, self).__init__()
        self.fc1 = nn.Linear(n, 5)
        self.fc2 = nn.Linear(5, 4)
        self.fc3 = nn.Linear(4, 1)

    def Forward(self, x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

