from torch import nn
import torch
import math

def positional_encoding(self, dim, max_len):

    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe

class PositionalEncodingLayer(nn.Module):
    def __init__(self, dim=512, max_len=5000):
        self.encoding = positional_encoding(dim, max_len)

    def forward(self, x):
        return x + self.encoding[:x.size(0), :]