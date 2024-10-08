import torch
from tqdm import tqdm

def model_eval(model, data_loader, criterion, device):
    model.eval()
    running_loss, correct = 0.0, 0.0
    total = len(data_loader.dataset)
    
    pbar = tqdm(data_loader)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    
    avg_loss = running_loss / total
    avg_accuracy = correct / total
    
    return avg_loss, avg_accuracy