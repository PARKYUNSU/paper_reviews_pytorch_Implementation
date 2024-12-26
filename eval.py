import torch

def evaluate_one_epoch(model, data_loader, loss_fn, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for texts, labels in data_loader:
            texts, labels = texts.to(device), labels.to(device)

            # Hidden and cell state 초기화
            hidden = model.init_hidden_and_cell_state(batch_size=len(texts), device=device)

            # Forward pass
            outputs = model(texts, hidden)
            loss = loss_fn(outputs, labels)

            # Accuracy 계산
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

    return running_loss / total, correct / total