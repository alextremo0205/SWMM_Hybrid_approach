import torch
import torch.nn as nn

class QInterchangeNN(nn.Module):
    def __init__(self):
        super(QInterchangeNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        dh = self.linear_relu_stack(x)
        return dh

class QRunoffNN(nn.Module):
    def __init__(self):
        super(QRunoffNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        dh = self.linear_relu_stack(x)
        return dh



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

