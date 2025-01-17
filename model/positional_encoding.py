import torch
import torch.nn as nn
import math

# positional_encoding.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_seq_len):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout_p)
 
        # (max_seq_len, dim_model) shape의 pos_encoding 생성
        pos_encoding = torch.zeros(max_seq_len, dim_model)
        positions_list = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_seq_len, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2, dtype=torch.float) * (-math.log(10000.0) / dim_model)
        )
 
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # pos_encoding: (max_seq_len, dim_model)
        # (1, max_seq_len, dim_model) 형태로 만들어서 batch 차원과 브로드캐스트할 때 편하도록 만듦
        pos_encoding = pos_encoding.unsqueeze(0)  # => (1, max_seq_len, dim_model)
        
        # register_buffer로 저장 (학습되지 않는 텐서)
        self.register_buffer("pos_encoding", pos_encoding)
 
    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """
        token_embedding: (batch_size, seq_len, d_model)
        """
        seq_len = token_embedding.size(1)  # 실제 동적인 seq_len

        # pos_encoding: (1, max_seq_len, d_model) => 필요 길이만큼 슬라이싱
        # => (1, seq_len, d_model)
        pe = self.pos_encoding[:, :seq_len, :]

        # 최종적으로 (batch_size, seq_len, d_model) + (1, seq_len, d_model)
        x = token_embedding + pe
        return self.dropout(x)
