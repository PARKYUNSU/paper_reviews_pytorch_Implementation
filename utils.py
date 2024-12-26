import torch
import torch.nn as nn
import torch.optim as optim
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        hx, cx = hidden
        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        return hy, cy

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(layer_dim)
        ])
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        batch_size, seq_length, _ = x.size()
        h, c = hidden

        for t in range(seq_length):
            input_t = x[:, t, :]
            for layer in range(self.layer_dim):
                h[layer], c[layer] = self.lstm_cells[layer](input_t, (h[layer], c[layer]))
                input_t = h[layer]

        out = self.dropout(h[-1])
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

    def init_hidden(self, batch_size, device):
        h = [torch.zeros(batch_size, self.hidden_dim).to(device) for _ in range(self.layer_dim)]
        c = [torch.zeros(batch_size, self.hidden_dim).to(device) for _ in range(self.layer_dim)]
        return h, c

def get_optimizer_and_criterion(model, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    return optimizer, criterion