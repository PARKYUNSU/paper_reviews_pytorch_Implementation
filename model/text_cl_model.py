import torch
import torch.nn as nn
import torch.nn.functional as F
from .lstm import LSTM
from torchtext.data.utils import get_tokenizer

class TextClassificationModel(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=True, dropout_prob=0.5, output_activation=None):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        self.lstm = LSTM(
            input_dim=embedding_dim,
            hidden_dim=hidden_size,
            layer_dim=num_layers,
            output_dim=num_classes,
            output_activation=output_activation,
            dropout_prob=dropout_prob,
        )

    def forward(self, x, hidden):
        x = self.embedding(x)
        return self.lstm(x, hidden)

    def init_hidden_and_cell_state(self, batch_size, device):
        """
        LSTM hidden state와 cell state 초기화.
        """
        return self.lstm.init_hidden(batch_size, device)