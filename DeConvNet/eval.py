import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets.squeeze(1).long())
            total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    print(f"Evaluation complete. Average Loss: {average_loss:.4f}")
    return average_loss
