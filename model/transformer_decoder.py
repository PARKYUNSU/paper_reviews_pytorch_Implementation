import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .decoder_layer import Decoder_Layer

# transformer_decoder.py

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)  # max_seq_len 전달
        self.layers = nn.ModuleList([Decoder_Layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        seq_len = tgt.size(1)  # Get the sequence length dynamically
        x = self.embedding(tgt) + self.positional_encoding(tgt)  # Ensure position encoding matches input length
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)

        logits = self.fc_out(x)
        return logits
