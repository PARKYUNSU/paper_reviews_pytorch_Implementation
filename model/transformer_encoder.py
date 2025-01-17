import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .encoder_layer import Encoder_Layer

# transformer_encoder.py

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)  # max_seq_len 전달
        self.layers = nn.ModuleList([Encoder_Layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        seq_len = x.size(1)  # Get the sequence length dynamically
        x = self.embedding(x) + self.positional_encoding(x)  # Ensure position encoding matches input length
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)
        return x