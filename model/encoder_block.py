import torch
import torch.nn as nn
from .attention import Attention
from .mlp import MLP

class Transformer_Encoder_Block(nn.Module):
    def __init__(self, config):
        super(Transformer_Encoder_Block, self).__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config, vis=True)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = MLP(config)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x