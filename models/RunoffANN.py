import torch
from torch import nn

class RunoffANN(nn.Module):
    def __init__(self):
        super(RunoffANN, self).__init__()
        
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.ReLU()
        )
    def forward(self, x):
        
        x = self.linear_stack(x)
        
        return x