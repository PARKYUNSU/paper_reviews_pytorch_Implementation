import torch
import torch.nn as nn
from torch.autograd import Variable
from .lstm_cell import LSTMCell

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm_cells = nn.ModuleList([LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, bias) for i in range(layer_dim)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = x.device
        
        # Initialize hidden and cell states for all layers
        h = [torch.zeros(x.size(0), self.hidden_dim, device=device) for _ in range(self.layer_dim)]
        c = [torch.zeros(x.size(0), self.hidden_dim, device=device) for _ in range(self.layer_dim)]
        
        outs = []

        # Iterate through each time step
        for seq in range(x.size(1)):
            input_t = x[:, seq, :]
            for i, cell in enumerate(self.lstm_cells):
                h[i], c[i] = cell(input_t, (h[i], c[i]))
                input_t = h[i]  # Output of current layer becomes input to next layer
            
            outs.append(h[-1])  # Store output of the last layer
        
        # Use output of the last time step for classification
        out = self.fc(outs[-1])
        return out