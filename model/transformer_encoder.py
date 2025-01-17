import torch
import torch.nn as nn
from . import PositionalEncoding
from .encoder_layer import Encoder_Layer

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)  # max_seq_len 전달
        self.layers = nn.ModuleList([Encoder_Layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len)
            mask: optional
        Returns:
            (batch_size, seq_len, d_model)
        """
        print("Encoder input shape:", x.shape)  # shape 확인 (batch_size, seq_len)

        # Embedding
        x = self.embedding(x)
        print("After embedding:", x.shape)      # shape 확인 (batch_size, seq_len, d_model)

        # Positional Encoding
        x = x + self.positional_encoding(x)
        print("After positional encoding:", x.shape)  # (batch_size, seq_len, d_model)

        # Dropout
        x = self.dropout(x)
        print("After dropout:", x.shape)        # (batch_size, seq_len, d_model)

        # Encoder Layers
        for i, layer in enumerate(self.layers, 1):
            x = layer(x, mask)
            print(f"After Encoder Layer {i}:", x.shape)

        return x