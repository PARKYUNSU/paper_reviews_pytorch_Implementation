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
    
#---#
# Debugging 
# # Model parameters
# input_dim = 10
# hidden_dim = 32
# layer_dim = 1
# output_dim = 5
# seq_len = 7
# batch_size = 4

# # Initialize the model
# model = LSTM(input_dim, hidden_dim, layer_dim, output_dim)

# # Test input
# x = torch.randn(batch_size, seq_len, input_dim)

# # Forward pass
# output = model(x)

# print("Input shape:", x.shape)
# print("Output shape:", output.shape)


# # Dummy labels for testing
# y = torch.randn(batch_size, output_dim)

# # Loss function
# criterion = nn.MSELoss()

# # Optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# # Forward pass
# output = model(x)

# # Calculate loss
# loss = criterion(output, y)
# print("Initial loss:", loss.item())

# # Backward pass
# optimizer.zero_grad()
# loss.backward()

# # Check gradients
# for name, param in model.named_parameters():
#     if param.grad is not None:
#         print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
#     else:
#         print(f"No gradient for {name}")


# # Extreme input values
# extreme_x = torch.randn(batch_size, seq_len, input_dim) * 1e6

# # Forward pass
# output = model(extreme_x)
# print("Output with extreme input values:", output)

# # Built-in LSTM
# builtin_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
# fc = nn.Linear(hidden_dim, output_dim)

# # Initialize hidden states
# h0 = torch.zeros(layer_dim, batch_size, hidden_dim)
# c0 = torch.zeros(layer_dim, batch_size, hidden_dim)

# # Forward pass
# builtin_output, _ = builtin_lstm(x, (h0, c0))
# builtin_output = fc(builtin_output[:, -1, :])  # Match our LSTM's output

# # Compare outputs
# our_output = model(x)
# print("Custom LSTM output:", our_output)
# print("Built-in LSTM output:", builtin_output)