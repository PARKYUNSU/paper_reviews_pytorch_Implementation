import torch
import torch.nn as nn
import torch.nn.functional as F

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Multi_Head_Attention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.depth = d_model // num_heads # 지정된 Head 수 만큼 나눔

        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dence = nn.Linear(d_model, d_model)
        self.value_dence = nn.Linear(d_model, d_model)
        self.output_dence = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        마지막 차원을 (num_heads, depth)로 분할
        이후 (batch_size, num_heads, seq_len, depth) 형태가 되도록 전치(Transpose)
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear layers for Q, K, V
        query = self.query_dense(query)  # (batch_size, seq_len, d_model)
        key = self.key_dense(key)        # (batch_size, seq_len, d_model)
        value = self.value_dense(value)  # (batch_size, seq_len, d_model)

        # Split heads
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len, depth)
        key = self.split_heads(key, batch_size)      # (batch_size, num_heads, seq_len, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len, depth)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.depth, dtype=torch.float32))
        if mask is not None:
            scores += mask * -1e9  # Apply mask to scores
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)  # (batch_size, num_heads, seq_len, depth)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.depth)
        return self.output_dense(attention_output)  # Final linear layer
    
class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFF, self).__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        x = self.ffnn(x)
        return x
    
class Encoder_Layer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder_Layer, self).__init__()
        