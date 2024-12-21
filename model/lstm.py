import torch
import torch.nn  as nn
import torch.nn.functional as F

from .lstm_cell import LSTMCell


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=device)


        outs = []
        cn =  c0[0,:,:]
        hn = h0[0,:,:]
       
        for seq in range(x.size(1)) :
           hn, cn = self.lstm(x[:, seq, :], (hn, cn))
           outs.append(hn)
           
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out
    
        
