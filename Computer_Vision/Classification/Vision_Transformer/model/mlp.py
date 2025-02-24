import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ACT2FN

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.act = ACT2FN["gelu"]
        self.fc2 = nn.Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x