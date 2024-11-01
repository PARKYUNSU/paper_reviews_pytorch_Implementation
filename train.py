def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(train_loader, 1):
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets.squeeze(1).long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 0:
            print(f"Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch complete. Average Loss: {epoch_loss:.4f}")
    return epoch_loss
