import torch
from torch import nn

class Sine(nn.Module):
    def forward(self, x):
        "Multiplies input by 30 before applying sine, as discussed in SIREN paper Sec. 3.2."
        return torch.sin(30 * x)

class AbsLayer(nn.Module):
    def __init__(self):
        super(AbsLayer, self).__init__()

    def forward(self, x):
        return torch.abs(x)