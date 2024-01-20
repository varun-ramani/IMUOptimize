from torch import nn
import torch
import math
from model.positional_encoding import PositionalEncodingLayer

class StepOneTransformer(nn.Module):
    def __init__(self):
        super(StepOneTransformer, self).__init__()

        num_sensors = 24
        transformer_dim = 512
        transformer_heads = 4
        transformer_layers = 6


        self.linear1 = nn.Linear(num_sensors * (3 + 3*3), transformer_dim)
        self.pos_encode = PositionalEncodingLayer(dim=transformer_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, 
            nhead=transformer_heads
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=6, 
            enable_nested_tensor=False
        )
        self.linear2 = nn.Linear(512, (24 * 9))

    def forward(self, x):
        x = self.linear1(x)
        x = self.pos_encode(x)
        x = self.transformer(x)
        x = self.linear2(x)
        return x