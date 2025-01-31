import torch
from model.vit import Vision_Transformer
from model.config import get_b16_config
from data import cifar_10
from tqdm import tqdm

def evaluate(pretrained_path, batch_size, device):
    config = get_b16_config()
    model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=False)
    
    # 모델을 CUDA로 이동
    model = model.to(device)
    
    # 모델 가중치 로드 (GPU/CPU 자동 매핑)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    # test_loader만 가져오기
    _, test_loader = cifar_10(batch_size)
    
    model.eval()
    correct = 0
    total = 0
    
    # tqdm을 사용하여 테스트 루프 진행 바 추가
    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating", unit="batch", ncols=100) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 진행 상황에 맞게 정확도 업데이트
                pbar.set_postfix(accuracy=f"{100 * correct / total:.2f}%")
    
    print(f'Final Accuracy: {100 * correct / total:.2f}%')

def main(pretrained_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device 설정
    evaluate(pretrained_path, batch_size, device)

if __name__ == "__main__":
    pass