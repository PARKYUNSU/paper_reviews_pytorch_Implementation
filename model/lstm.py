import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM 레이어
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout if layer_dim > 1 else 0)

        # 완전 연결층
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 임베딩 레이어 통과
        x = self.embedding(x)

        # hidden state와 cell state 초기화
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)

        # LSTM 순전파
        out, _ = self.lstm(x, (h0, c0))

        # 마지막 시간 단계 출력 반환
        out = self.fc(out[:, -1, :])  # 출력 크기: (batch_size, output_dim)
        return out
