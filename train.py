# train.py

import torch
import torch.nn as nn
import torch.optim as optim

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    모델, dataloader, loss 함수, optimizer, device를 받아
    한 번의 epoch 훈련을 수행하는 예시 함수.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (src, tgt) in enumerate(dataloader):
        src = src.to(device)  # (batch_size, seq_len)
        tgt = tgt.to(device)  # (batch_size, seq_len)
        
        # 타겟을 <start_token>, <end_token> 등으로 분리해서 쓰는 경우도 많으나,
        # 여기서는 간단히 tgt = src 복사 형태이므로 그대로 사용
        # 실제로는 tgt_in, tgt_out 분리 로직 필요할 수 있음
        tgt_in = tgt[:, :-1]  # 마지막 토큰 제외한 입력
        tgt_out = tgt[:, 1:]  # 첫 번째 토큰 제외한 정답
        
        # 마스크 생성
        # 여기서는 간단하게 사용하지만, 프로젝트 요구사항에 맞춰 수정
        from data import create_mask
        src_mask, tgt_mask, memory_mask = create_mask(src, tgt_in, pad_idx=0)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        memory_mask = memory_mask.to(device)

        # 모델 forward
        outputs = model(src, tgt_in, tgt_mask=tgt_mask, memory_mask=memory_mask)
        # outputs shape: (batch_size, seq_len-1, vocab_size)

        # Loss 계산
        # CrossEntropyLoss를 위해 view를 변환
        outputs_reshaped = outputs.view(-1, outputs.size(-1))  # (batch_size*(seq_len-1), vocab_size)
        tgt_out_reshaped = tgt_out.reshape(-1)                 # (batch_size*(seq_len-1))

        loss = criterion(outputs_reshaped, tgt_out_reshaped)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss