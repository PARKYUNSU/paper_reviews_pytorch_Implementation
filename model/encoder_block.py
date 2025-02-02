import torch
import torch.nn as nn
from .attention import Attention
from .mlp import MLP

class Transformer_Encoder_Block(nn.Module):
    def __init__(self, config):
        super(Transformer_Encoder_Block, self).__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        # 훈련 및 평가 시에는 vis=False로 하는 것이 좋습니다.
        self.attn = Attention(config, vis=True)  # 시각화 모드를 사용하려면 True로 두되...
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.mlp = MLP(config)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        # 만약 attn_out이 튜플이면 (output, attention_probs)가 반환된 것이므로 output만 사용합니다.
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x