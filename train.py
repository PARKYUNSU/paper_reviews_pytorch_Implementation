import torch
import torch.nn as nn
import torch.optim as optim
from data import create_mask

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)  # (batch_size, seq_len)
        tgt = tgt.to(device)  # (batch_size, seq_len)
        
        tgt_in = tgt[:, :-1]  # 마지막 토큰 제외한 입력
        tgt_out = tgt[:, 1:]  # 첫 번째 토큰 제외한 정답
        
        
        src_mask, tgt_mask, memory_mask = create_mask(src, tgt_in, pad_idx=0)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        memory_mask = memory_mask.to(device)

        outputs = model(src, tgt_in, tgt_mask=tgt_mask, memory_mask=memory_mask) # (batch_size, seq_len-1, vocab_size)
        outputs_reshaped = outputs.view(-1, outputs.size(-1))  # (batch_size*(seq_len-1), vocab_size)
        tgt_out_reshaped = tgt_out.reshape(-1)                 # (batch_size*(seq_len-1))

        loss = criterion(outputs_reshaped, tgt_out_reshaped)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss