import torch
import torch.nn as nn
    
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