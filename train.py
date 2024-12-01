import torch
import torch_xla.core.xla_model as xm

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)

        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)  # TPU-specific optimizer step

        # Update metrics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy