import torch
from tqdm import tqdm

def evaluate(model, val_loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    return avg_loss, accuracy