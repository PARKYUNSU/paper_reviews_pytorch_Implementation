import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        assert config.hidden_size % config.transformer["num_heads"] == 0, "hiddend_size must be divisible by num_heads"
        self.num_heads = config.transformer["num_heads"]
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.all_head_dim = self.num_heads * self.head_dim

        #self.qkv = nn.Linear(self.hidden_size, self.all_head_dim* 3) # 한 번에 qkv 연산
        self.query_dense = nn.Linear(self.hidden_size, self.all_head_dim)
        self.key_dense = nn.Linear(self.hidden_size, self.all_head_dim)
        self.value_dense = nn.Linear(self.hidden_size, self.all_head_dim)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.output_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj_dropout = nn.Linear(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):

        x = 