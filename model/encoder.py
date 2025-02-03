import torch
import torch.nn as nn
from .encoder_block import Transformer_Encoder_Block

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([Transformer_Encoder_Block(config) for _ in range(config.transformer["num_layers"])])
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x