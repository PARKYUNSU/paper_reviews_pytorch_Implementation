import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_seq_len):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
 
        # Encoding - From formula
        pos_encoding = torch.zeros(max_seq_len, dim_model)
        positions_list = torch.arange(0, max_seq_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
 
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # pos_encoding shape: (max_seq_len, dim_model)

        # (1, max_seq_len, dim_model) 형태로 만들어서 batch 차원과 broadcast 가능하게 만듦
        pos_encoding = pos_encoding.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pos_encoding", pos_encoding)
 
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        """
        token_embedding: (batch_size, seq_len, d_model)
        """
        seq_len = token_embedding.size(1)
        
        # pos_encoding: (1, max_seq_len, d_model)
        # 필요 길이만큼 슬라이싱: (1, seq_len, d_model)
        pos_encoding = self.pos_encoding[:, :seq_len, :]

        # 최종 덧셈 시: (batch_size, seq_len, d_model) + (1, seq_len, d_model)
        return self.dropout(token_embedding + pos_encoding)