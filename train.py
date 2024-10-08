import torch
from tqdm import tqdm

def model_train(model, data_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss, correct = 0.0, 0.0
    total = len(data_loader.dataset)
    
    pbar = tqdm(data_loader)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    
    avg_loss = running_loss / total
    avg_accuracy = correct / total
    
    return avg_loss, avg_accuracy