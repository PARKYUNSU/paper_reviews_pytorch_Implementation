import time
import torch

def train(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for i, (images, targets) in enumerate(train_loader, 1):
        images, targets = images.to(device), targets.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, targets.squeeze(1).long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_batch = elapsed_time / i
            remaining_batches = len(train_loader) - i
            eta = avg_time_per_batch * remaining_batches

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}, ETA: {eta:.2f} seconds")

    epoch_loss = running_loss / len(train_loader)
    total_time = time.time() - start_time
    print(f"Epoch [{epoch+1}/{num_epochs}] complete. Average Loss: {epoch_loss:.4f}, Time Taken: {total_time:.2f} seconds")
    return epoch_loss
