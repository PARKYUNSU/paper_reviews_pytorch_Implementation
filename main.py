import torch
from torch.utils.data import DataLoader
from model import LSTM
from .data import generate_data, SineWaveDataset
from train import train_model
from eval import evaluate_model, plot_training_curves

# 데이터 생성
seq_length = 50
num_samples = 1000
data = generate_data(seq_length, num_samples)
train_data = data[:800]
test_data = data[800:]

# 데이터셋 및 DataLoader 준비
train_dataset = SineWaveDataset(train_data)
test_dataset = SineWaveDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 모델 초기화
input_dim = 1
hidden_dim = 64
layer_dim = 1
output_dim = 1
model = LSTM(input_dim, hidden_dim, layer_dim, output_dim)

# 손실 함수 및 옵티마이저
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
num_epochs = 20
train_losses = train_model(train_loader, model, criterion, optimizer, num_epochs)

# 학습 곡선 시각화
plot_training_curves(train_losses)

# 평가
evaluate_model(test_loader, model)