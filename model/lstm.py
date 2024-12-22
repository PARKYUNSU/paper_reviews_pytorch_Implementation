import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0.2):
        """
        Initialize the LSTM model.
        Args:
            input_dim (int): Size of the input features.
            hidden_dim (int): Number of features in the hidden state.
            layer_dim (int): Number of recurrent layers.
            output_dim (int): Size of the output features.
            dropout (float): Dropout probability for regularization.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, 
                            batch_first=True, dropout=dropout if layer_dim > 1 else 0)

        # Fully connected layer for final predictions
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_sequences=False):
        """
        Forward pass for the LSTM model.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            return_sequences (bool): If True, return outputs for all time steps.
        Returns:
            Tensor: Final output tensor.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode either the last time step or the full sequence
        if return_sequences:
            return self.fc(out)  # Shape: (batch_size, seq_len, output_dim)
        else:
            return self.fc(out[:, -1, :])  # Shape: (batch_size, output_dim)
