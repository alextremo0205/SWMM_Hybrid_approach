import torch
from torch import nn

class InterchangeANN(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(InterchangeANN, self).__init__()

        self.linear_stack = nn.Sequential(

            nn.Linear(in_dims, 8, bias = False),
            nn.ReLU(),
            nn.Linear(8, 4, bias = False),
            nn.ReLU(),
            nn.Linear(4, out_dims, bias = False),
            
            nn.ReLU()
        )
    def forward(self, x):

        x = self.linear_stack(x)

        return x
