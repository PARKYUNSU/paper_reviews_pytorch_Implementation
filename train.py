from tqdm import tqdm
import torch

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(data_loader, unit='batch', total=len(data_loader))
    
    for texts, labels in progress_bar:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()

        model.init_hidden_and_cell_state(len(texts), device)
        outputs = model(texts)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        progress_bar.set_description(f"Train Loss: {running_loss/(total):.4f}, Accuracy: {correct/total:.4f}")

    return running_loss / total, correct / total