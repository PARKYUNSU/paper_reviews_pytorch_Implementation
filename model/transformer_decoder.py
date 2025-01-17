import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .decoder_layer import Decoder_Layer

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        self.layers = nn.ModuleList([Decoder_Layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        # Embedding
        x = self.embedding(tgt)
        # Positional Encoding
        x = x + self.positional_encoding(x)
        x = self.dropout(x)
        # Decoder Layers
        for layer in enumerate(self.layers, 1):
            x = layer(x, enc_output, tgt_mask, memory_mask)

        # Final Linear
        logits = self.fc_out(x)

        return logits