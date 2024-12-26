import torch
import torch.nn as nn
import torch.nn.functional as F
from model.lstm import LSTM
from torchtext.data.utils import get_tokenizer

class TextClassificationModel(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=True, dropout_prob=0.5):
        """
        Args:
            num_classes (int): 클래스 개수
            vocab_size (int): 단어 사전 크기
            embedding_dim (int): 임베딩 차원
            hidden_size (int): LSTM hidden state 크기
            num_layers (int): LSTM 레이어 개수
            bidirectional (bool): 양방향 여부
            dropout_prob (float): 드롭아웃 확률
        """
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(embedding_dim, hidden_size, num_layers, num_classes, dropout_prob=dropout_prob)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_classes = num_classes

    def forward(self, x, hidden):
        x = self.embedding(x)
        return self.lstm(x, hidden)

    def init_hidden(self, batch_size, device):
        return self.lstm.init_hidden(batch_size, device)
    
def get_basic_tokenizer():
    """
    기본 토크나이저를 반환합니다.
    Returns:
        Callable: 텍스트를 토큰 단위로 나누는 함수
    """
    return get_tokenizer("basic_english")