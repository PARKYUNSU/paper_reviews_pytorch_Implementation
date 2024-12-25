import torch
from torch.utils.data import DataLoader
from model.lstm import LSTM
from data.generate_data import prepare_imdb_data  # IMDB 데이터 준비 함수
from train import train_model
from utils import plot_training_curves
from eval import evaluate_model  # 평가 함수 가져오기

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# IMDB 데이터 로드 및 준비
csv_path = "IMDB Dataset.csv"
seq_length = 500  # 고정된 시퀀스 길이
train_dataset, val_dataset, vocab = prepare_imdb_data(csv_path, seq_length)

# DataLoader 생성
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 초기화
input_dim = len(vocab) + 1  # 단어 사전 크기 (패딩을 위한 추가 값 포함)
hidden_dim = 256
layer_dim = 2  # LSTM 레이어 수
output_dim = 1
model = LSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# 손실 함수 및 옵티마이저
criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
num_epochs = 10
train_losses, val_losses, train_acc, val_acc = train_model(
    train_loader, val_loader, model, criterion, optimizer, num_epochs, device
)

# 학습 곡선 시각화
plot_training_curves(train_losses, val_losses, save_path="training_loss_curve.png")

# 모델 평가 및 결과 시각화
evaluate_model(val_loader, model, device, save_path="prediction_graph.png")