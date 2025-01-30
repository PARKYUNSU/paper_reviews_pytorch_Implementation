import torch
from model.vit import Vision_Transformer
from model.config import get_b16_config
from data import cifar_10

def evaluate(pretrained_path, batch_size, device):
    config = get_b16_config()
    model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=False)
    
    # 모델을 CUDA로 이동
    model = model.to(device)
    
    model.load_state_dict(torch.load(pretrained_path))
    
    _, test_loader = cifar_10(batch_size)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터도 CUDA로 이동
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')

def main(pretrained_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device 설정
    evaluate(pretrained_path, batch_size, device)

if __name__ == "__main__":
    pass