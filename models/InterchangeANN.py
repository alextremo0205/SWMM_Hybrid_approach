import torch
from torch import nn

class InterchangeANN(nn.Module):
    def __init__(self):
        super(InterchangeANN, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Tanh()
        )
    def forward(self, x):
        
        x = self.linear_relu_stack(x)
        
        return x
    
    