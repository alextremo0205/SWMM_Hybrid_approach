import torch
from torch import nn

class NodeFeaturesANN(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(NodeFeaturesANN, self).__init__()
        
        self.linear_stack = nn.Sequential(
            nn.Linear(in_dims, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, out_dims),
            nn.ReLU()
        )
    def forward(self, x):
        
        x = self.linear_stack(x)
        
        return x