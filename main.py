import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR  # 학습률 스케줄러 추가
from model.lstm import LSTM
from data.generate_data import prepare_stock_data, SineWaveDataset
from data.generate_data import download_stock_data, save_stock_data
from train import train_model
from eval import evaluate_model, plot_training_curves

# 주식 데이터 다운로드 및 저장
ticker = "AAPL"  # 애플 주식
start_date = "2020-01-01"
end_date = "2023-12-31"
file_path = "apple_stock_data.csv"

stock_data = download_stock_data(ticker, start_date, end_date)
save_stock_data(stock_data, file_path)

# 주식 데이터를 시계열로 변환
seq_length = 50
data = prepare_stock_data(file_path, seq_length)
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

# 학습률 스케줄러 추가
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # 50 에포크마다 학습률 감소

# 학습
num_epochs = 500
train_losses = []

for epoch in range(num_epochs):
    # train_model 함수는 단일 에포크의 평균 손실을 반환합니다.
    epoch_loss = train_model(train_loader, model, criterion, optimizer, num_epochs=1)
    train_losses.append(epoch_loss)

    # 출력
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 학습 곡선 시각화 및 저장
plot_training_curves(train_losses, save_path="training_loss_curve.png")

# 평가 그래프 저장
evaluate_model(test_loader, model, save_path="prediction_graph.png")