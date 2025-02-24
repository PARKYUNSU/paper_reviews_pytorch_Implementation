import torch
import torch.nn as nn
from data import create_mask

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)  
        tgt = tgt.to(device)  
        
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        src_mask, tgt_mask, memory_mask = create_mask(src, tgt_in, pad_idx=0)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        memory_mask = memory_mask.to(device)

        outputs = model(src, tgt_in, tgt_mask=tgt_mask, memory_mask=memory_mask)
        outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
        tgt_out_reshaped = tgt_out.reshape(-1)

        loss = criterion(outputs_reshaped, tgt_out_reshaped)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask, tgt_mask, memory_mask = create_mask(src, tgt_in, pad_idx=0)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            memory_mask = memory_mask.to(device)

            outputs = model(src, tgt_in, tgt_mask=tgt_mask, memory_mask=memory_mask)
            outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
            tgt_out_reshaped = tgt_out.reshape(-1)

            loss = criterion(outputs_reshaped, tgt_out_reshaped)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss