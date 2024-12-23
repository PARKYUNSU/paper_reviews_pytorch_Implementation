import torch
import torch.nn as nn
from torch.autograd import Variable
from .lstm_cell import LSTMCell

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim, hidden_dim, bias)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h0 = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        hn, cn = h0, c0
        for t in range(seq_len):
            hn, cn = self.lstm(x[:, t, :], (hn, cn))

        out = self.fc(hn)
        return out

