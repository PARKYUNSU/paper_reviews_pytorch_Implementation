import torch
import torch.nn as nn
from .lstm_cell import LSTMCell

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.5):
        """
        커스텀 LSTM 클래스
        Args:
            input_dim (int): 입력 특성의 차원
            hidden_dim (int): 각 LSTM 레이어의 hidden state 크기
            layer_dim (int): LSTM 레이어 수
            output_dim (int): 최종 출력 크기
            dropout_prob (float): Dropout 확률
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # 여러 LSTM 레이어를 위한 LSTMCell 배열
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(layer_dim)
        ])
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        """
        Args:
            x: 입력 시퀀스, [batch_size, seq_length, input_dim]
            hidden: 초기 hidden state와 cell state (h, c)

        Returns:
            out: 모델 출력, [batch_size, output_dim]
        """
        batch_size, seq_length, _ = x.size()
        h, c = hidden  # 초기 hidden state와 cell state 분리

        # 시간 축을 따라 순환
        for t in range(seq_length):
            input_t = x[:, t, :]  # 현재 타임스텝 입력
            for layer in range(self.layer_dim):
                h[layer], c[layer] = self.lstm_cells[layer](input_t, (h[layer], c[layer]))
                input_t = h[layer]  # 다음 레이어로 전달

        # 마지막 타임스텝의 출력 사용
        out = self.dropout(h[-1])  # 가장 마지막 레이어의 hidden state
        out = self.fc(out)  # 출력 레이어 통과
        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        elif self.output_activation == "softmax":
            out = F.softmax(out, dim=-1)
        return out

    def init_hidden(self, batch_size, device):
        """
        hidden state와 cell state 초기화
        Args:
            batch_size (int): 배치 크기
            device (torch.device): 사용 중인 장치 (CPU 또는 CUDA)

        Returns:
            tuple: (h, c) 초기 hidden state와 cell state
        """
        h = [torch.zeros(batch_size, self.hidden_dim).to(device) for _ in range(self.layer_dim)]
        c = [torch.zeros(batch_size, self.hidden_dim).to(device) for _ in range(self.layer_dim)]
        return h, c