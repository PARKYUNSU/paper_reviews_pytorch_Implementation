import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .decoder_layer import Decoder_Layer

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout=0.1):
        """
        Transformer Decoder
        Args:
            num_layers: Number of decoder layers
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden layer size in FFNN
            vocab_size: Vocabulary size
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([Decoder_Layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        """
        Forward pass of the decoder
        Args:
            tgt: Target sequence (batch_size, tgt_seq_len)
            enc_output: Encoder output (batch_size, src_seq_len, d_model)
            tgt_mask: Mask for the target sequence
            memory_mask: Mask for the encoder-decoder attention
        """
        seq_len = tgt.size(1)
        x = self.embedding(tgt) + self.positional_encoding(tgt)
        x = self.dropout(x)

        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)

        # Project to vocabulary size
        logits = self.fc_out(x)
        return logits