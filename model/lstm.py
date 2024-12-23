import torch
import torch.nn as nn
from .lstm_cell import LSTMCell

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm_cells = nn.ModuleList(
            [LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(layer_dim)]
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)

        # Initialize hidden and cell states
        h = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.layer_dim)]
        c = [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in range(self.layer_dim)]

        # Process sequence
        for t in range(x.size(1)):  # seq_len
            input_t = x[:, t, :]
            for i, cell in enumerate(self.lstm_cells):
                h[i], c[i] = cell(input_t, (h[i], c[i]))
                input_t = h[i]  # Pass hidden state to the next layer

        # Final hidden state from the last layer
        out = self.fc(h[-1])
        return out