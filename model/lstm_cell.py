import torch
import torch.nn  as nn
import torch.nn.functional as F
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, dropout_prob=0.2):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dropout = nn.Dropout(dropout_prob)
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x2h.weight)
        nn.init.xavier_uniform_(self.h2h.weight)
        if self.x2h.bias is not None:
            nn.init.zeros_(self.x2h.bias)
            nn.init.zeros_(self.h2h.bias)

    def forward(self, x, hidden):
        hx, cx = hidden
        gates = self.x2h(x) + self.h2h(self.dropout(hx))
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        return hy, cy