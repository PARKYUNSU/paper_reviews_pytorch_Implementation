import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_head_attention import Multi_Head_Attention
from .positionwise_ffnn import PositionwiseFF

class Encoder_Layer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder_Layer, self).__init__()
        self.self_attention = Multi_Head_Attention(d_model, num_heads)
        self.ffnn = PositionwiseFF(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model) # layer Normalization
        self.norm2 = nn.LayerNorm(d_model) # layer Normalization
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attention_output))

        ffnn_output = self.ffnn(x)
        x = self.norm2(x + self.dropout2(ffnn_output))
        return x