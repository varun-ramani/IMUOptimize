from torch import nn
import torch
import math

class StepOneTransformer(nn.Module):
    def __init__(self, num_hidden=1024, num_layers=2, nhead=8):
        super(StepOneTransformer, self).__init__()
        self.linear = nn.Linear(24 * (3 + 9), num_hidden)
        transformer_layer = nn.TransformerEncoderLayer(d_model=num_hidden, nhead=nhead)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.linear2 = nn.Linear(num_hidden, (24 * 9))
        self.positional_encoding = self._generate_positional_encoding(num_hidden)

    def _generate_positional_encoding(self, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x):
        x = self.linear(x)
        x = x + self.positional_encoding[:x.size(0), :]
        x = self.transformer(x)
        x = self.linear2(x)
        return x

