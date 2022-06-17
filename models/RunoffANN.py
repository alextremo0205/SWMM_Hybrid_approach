import torch
from torch import nn

class RunoffANN(nn.Module):
    def __init__(self):
        super(RunoffANN, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.Sigmoid(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        
        x = self.linear_relu_stack(x)
        
        return x