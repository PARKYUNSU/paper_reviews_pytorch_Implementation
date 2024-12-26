import torch

def evaluate_one_epoch(model, data_loader, loss_fn, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)

            model.init_hidden_and_cell_state(len(texts), device)
            outputs = model(texts)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total