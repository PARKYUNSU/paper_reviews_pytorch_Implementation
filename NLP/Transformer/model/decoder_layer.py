import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_head_attention import Multi_Head_Attention
from .positionwise_ffnn import PositionwiseFF

class Decoder_Layer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(Decoder_Layer, self).__init__()
        
        # Masked Mutil Head Attention
        self.self_attention = Multi_Head_Attention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # Encoder-Decoder Attention
        self.enc_dec_attention = Multi_Head_Attention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        # Position-wise Feed Foward Network
        self.ffnn = PositionwiseFF(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # Masked Multi Head Attention
        self_attn_output = self.self_attention(x, x, x, mask=tgt_mask)  # tgt_mask 사용
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Encoder-Decoder Attention
        enc_dec_attn_output = self.enc_dec_attention(x, enc_output, enc_output, mask=memory_mask)
        x = self.norm2(x + self.dropout2(enc_dec_attn_output))

        # Feed-forward Network
        ffn_output = self.ffnn(x)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x