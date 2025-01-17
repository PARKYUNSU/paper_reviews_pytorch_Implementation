import torch
from model.transformer import Transformer
from data import generate_random_data, batchify_data

device = "cuda" if torch.cuda.is_available() else "cpu"

# 하이퍼파라미터
num_layers = 6
d_model = 512
num_heads = 8
d_ff = 2048
vocab_size = 10000
max_seq_len = 100
dropout = 0.1

train_data = generate_random_data(9000)
val_data = generate_random_data(3000)

train_dataloader = batchify_data(train_data)
val_dataloader = batchify_data(val_data)

# 모델, 옵티마이저, 손실 함수 설정
model = Transformer(num_tokens=4, dim_model=d_model, num_heads=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout_p=dropout).to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# 모델 훈련
train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs=10)