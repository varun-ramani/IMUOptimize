from torch import nn
import torch
import math
import utils

class PositionalEncodingLayer(nn.Module):
    def __init__(self, dim=512, max_len=5000):
        super().__init__()

        posenc = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        posenc[:, 0::2] = torch.sin(position * div_term)
        posenc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('posenc', posenc)

    def forward(self, x):
        return x + self.posenc[:x.size(0), :]