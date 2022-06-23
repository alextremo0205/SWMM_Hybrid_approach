import torch
from torch import nn

class InterchangeANN(nn.Module):
    def __init__(self):
        super(InterchangeANN, self).__init__()
        
        self.linear_stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU()
        )
    def forward(self, x):
        
        x = self.linear_stack(x)
        
        return x
    
    