import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding
from .decoder_layer import Decoder_Layer

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)  # max_seq_len 전달
        self.layers = nn.ModuleList([Decoder_Layer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: (batch_size, tgt_seq_len)
            enc_output: (batch_size, src_seq_len, d_model)
        Returns:
            logits: (batch_size, tgt_seq_len, vocab_size)
        """
        print("Decoder input shape (tgt):", tgt.shape)  
        print("Encoder output shape (enc_output):", enc_output.shape)

        # Embedding
        x = self.embedding(tgt)
        print("Decoder after embedding:", x.shape)  # (batch_size, tgt_seq_len, d_model)

        # Positional Encoding
        x = x + self.positional_encoding(x)
        print("Decoder after positional encoding:", x.shape)  # (batch_size, tgt_seq_len, d_model)

        # Dropout
        x = self.dropout(x)
        print("Decoder after dropout:", x.shape)  # (batch_size, tgt_seq_len, d_model)

        # Decoder Layers
        for i, layer in enumerate(self.layers, 1):
            x = layer(x, enc_output, tgt_mask, memory_mask)
            print(f"After Decoder Layer {i}:", x.shape)

        # Final Linear
        logits = self.fc_out(x)
        print("Decoder final logits shape:", logits.shape)  # (batch_size, tgt_seq_len, vocab_size)

        return logits