import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def evaluate_model(test_loader, model, device, save_path=None):
    """
    모델 평가 함수: 테스트 데이터를 사용해 모델의 예측값과 실제값을 비교하고 시각화.

    Args:
        test_loader (DataLoader): 테스트 데이터 로더.
        model (nn.Module): 평가할 학습된 모델.
        device (torch.device): 모델과 데이터를 실행할 디바이스 (CPU/GPU).
        save_path (str, optional): 그래프를 저장할 경로. 기본값은 None.

    Returns:
        None
    """
    model.eval()  # 평가 모드로 전환
    predictions = []
    actuals = []

    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating", unit="batch") as tbar:
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)

                # 데이터 타입 변환
                sequences = sequences.float()
                
                # 입력 데이터에 추가 차원을 삽입 (batch_size, seq_length, 1)
                sequences = sequences.unsqueeze(-1)

                # 모델 예측
                outputs = model(sequences)
                predictions.extend(outputs.squeeze().tolist())
                actuals.extend(labels.squeeze().tolist())


    # 예측 결과 vs 실제 값 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual", linestyle="--", alpha=0.7)
    plt.plot(predictions, label="Predicted", alpha=0.9)
    plt.legend()
    plt.title("Actual vs Predicted (Test Data)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")

    if save_path:  # 저장 경로가 제공된 경우
        plt.savefig(save_path)
        print(f"Prediction graph saved to {save_path}")
    else:  # 제공되지 않은 경우 시각화
        plt.show()