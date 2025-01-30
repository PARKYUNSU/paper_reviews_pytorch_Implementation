import torch
from model.vit import Vision_Transformer
from model.config import get_b16_config
from data import cifar_10
from transformers import AutoModelForImageClassification

def evaluate(pretrained_path, batch_size):
    # config = get_b16_config()
    # model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=False)
    model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

    
    model.load_state_dict(torch.load(pretrained_path))
    
    _, test_loader = cifar_10(batch_size)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')

def main(pretrained_path, batch_size):
    evaluate(pretrained_path, batch_size)

if __name__ == "__main__":
    pass