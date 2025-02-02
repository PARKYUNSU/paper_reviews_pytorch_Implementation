import torch
from model.vit import Vision_Transformer
from model.config import get_b16_config
from data import cifar_10
from tqdm import tqdm

def evaluate(pretrained_path, batch_size, device):
    config = get_b16_config()
    model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=False)
    model.load_state_dict(torch.load("fine_tuned_model.pth", map_location=device))
    model = model.to(device)
    
    pretrained_weights = torch.load(pretrained_path, map_location=device)
    model.load_from(pretrained_weights)
    
    _, test_loader = cifar_10(batch_size)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating", unit="batch", ncols=100) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix(accuracy=f"{100 * correct / total:.2f}%")
    
    print(f'Final Accuracy: {100 * correct / total:.2f}%')

def main(pretrained_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device 설정
    evaluate(pretrained_path, batch_size, device)

if __name__ == "__main__":
    pass