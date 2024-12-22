import torch
import torch.nn as nn
from model.lstm import LSTM
from data.generate_data import get_dataloaders
from utils import load_checkpoint

# eval.py에서 수정
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 테스트 데이터 로드
    _, test_loader, vocab_size = get_dataloaders(data_dir="data", batch_size=32)

    # 모델 초기화
    input_dim = vocab_size
    hidden_dim = 128
    layer_dim = 2
    output_dim = 1  # 이진 분류이므로 output_dim = 1
    model = LSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    
    # 모델 체크포인트 로드
    _, _ = load_checkpoint(model, None)  # 모델 가중치만 로드
    
    # 평가
    model.eval()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()  # 이진 분류에 적합한 Loss 사용

    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs).squeeze()  # 출력 차원을 맞추기 위해 squeeze() 추가
            loss = criterion(outputs, targets.float())
            total_loss += loss.item()

            # Accuracy 계산
            predictions = (torch.sigmoid(outputs) >= 0.5).long()  # 0.5 이상이면 1로 분류
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples * 100
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.2f}%")

    # 평가 결과 저장
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Evaluation Loss: {avg_loss:.4f}\n")
        f.write(f"Evaluation Accuracy: {accuracy:.2f}%\n")

    return avg_loss, accuracy

if __name__ == "__main__":
    avg_loss, accuracy = evaluate()