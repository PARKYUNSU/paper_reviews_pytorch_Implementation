import torch
import torch.nn as nn
from torch.autograd import Variable
from .lstm_cell import LSTMCell

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # 여러 LSTM 레이어를 위한 LSTMCell 배열
        self.lstm_cells = nn.ModuleList([LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) 
                                         for i in range(layer_dim)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 초기 hidden state와 cell state 정의
        h = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.layer_dim)]
        c = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.layer_dim)]

        # 시퀀스를 시간축으로 처리
        for t in range(seq_len):
            input_t = x[:, t, :]  # 현재 타임스텝 입력
            for layer in range(self.layer_dim):
                h[layer], c[layer] = self.lstm_cells[layer](input_t, (h[layer], c[layer]))
                input_t = h[layer]  # 다음 레이어로 전달

        # 마지막 타임스텝의 출력으로 최종 예측 수행
        out = self.fc(h[-1])
        return out