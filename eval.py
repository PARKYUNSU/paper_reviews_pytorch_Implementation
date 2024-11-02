import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    corrects = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets.squeeze(1).long())
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, dim=1)
            corrects += torch.sum(preds == targets.squeeze(1))
            total_samples += targets.size(0)
    
    average_loss = total_loss / len(dataloader)
    accuracy = (corrects / total_samples).item() * 100
    print(f"Evaluation complete. Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return average_loss, accuracy
