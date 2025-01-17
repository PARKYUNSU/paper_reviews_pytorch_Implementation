import torch
import torch.nn as nn
from model.transformer import Transformer
from data import generate_random_data, batchify_data
from train import fit

device = "cuda" if torch.cuda.is_available() else "cpu"

# 하이퍼파라미터 설정
num_layers = 3
d_model = 8
num_heads = 2
d_ff = 16
vocab_size = 4  # 실제로는 vocab_size가 이보다 클 수 있습니다.
max_seq_len = 8
dropout = 0.1

# 학습 데이터 생성 및 배치화
train_data = generate_random_data(9000)
val_data = generate_random_data(3000)

train_dataloader = batchify_data(train_data)
val_dataloader = batchify_data(val_data)

# 모델 인스턴스화
model = Transformer(num_layers=num_layers, d_model=d_model, num_heads=num_heads, d_ff=d_ff, vocab_size=vocab_size, max_seq_len=max_seq_len, dropout=dropout).to(device)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)  # Xavier 초기화
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # 바이어스는 0으로 초기화

model.apply(init_weights)


# 옵티마이저와 손실 함수 설정
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# 모델 훈련
train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs=10, device=device)