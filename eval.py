import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            hidden = model.init_hidden(inputs.size(0), device)
            outputs = model(inputs, hidden).squeeze()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return test_loss / len(dataloader), accuracy