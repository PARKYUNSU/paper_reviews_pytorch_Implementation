import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout=0.2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM 레이어 정의
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, 
                            batch_first=True, dropout=dropout if layer_dim > 1 else 0)

        # 최종 예측을 위한 완전 연결층
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: 입력 텐서 (batch_size, seq_len, input_dim)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # hidden state와 cell state 초기화
        batch_size = x.size(0)  # 배치 크기 추출
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(device)

        # LSTM 순전파
        out, _ = self.lstm(x, (h0, c0))

        # 마지막 시간 단계 출력 반환
        out = self.fc(out[:, -1, :])  # 출력 크기: (batch_size, output_dim)
        return out
