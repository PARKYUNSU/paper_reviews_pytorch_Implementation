import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder

class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_seq_len, dropout)

    def forward(self, src, tgt, tgt_mask=None, memory_mask=None):
        enc_output = self.encoder(src)
        output = self.decoder(tgt, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output