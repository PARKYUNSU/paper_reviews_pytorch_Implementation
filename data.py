import torch
from torch.utils.data import Dataset, DataLoader
import random

class DummyDataset(Dataset):
    """
    임의의 숫자 시퀀스를 만들어 (src, tgt)로 반환하는 예시 Dataset.
    """
    def __init__(self, total_samples=1000, seq_len=10, vocab_size=100):
        super().__init__()
        self.total_samples = total_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # [seq_len] 길이의 임의의 입력 시퀀스
        src = [random.randint(1, self.vocab_size - 1) for _ in range(self.seq_len)]
        # 간단하게 tgt = src 를 복사하는 형태로 예시
        tgt = src[:]  # 실제로는 다양한 transform이 들어갈 수 있음
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch, pad_idx=0):
    src_list, tgt_list = zip(*batch)

    # 길이가 모두 동일하므로 그냥 stack
    src = torch.stack(src_list, dim=0)  # (batch_size, seq_len)
    tgt = torch.stack(tgt_list, dim=0)  # (batch_size, seq_len)

    return src, tgt

def create_mask(src, tgt, pad_idx=0):
    # Padding Mask 생성 예시
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    tgt_pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    
    # Look Ahead Mask (Decoder용)
    seq_len = tgt.size(1)
    subsequent_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()  # (seq_len, seq_len)
    subsequent_mask = subsequent_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)

    tgt_mask = tgt_pad_mask | subsequent_mask.to(tgt.device)
    memory_mask = src_mask

    return src_mask, tgt_mask, memory_mask

def get_dataloader(batch_size=32, seq_len=10, vocab_size=100, total_samples=1000):
    dataset = DummyDataset(total_samples=total_samples, seq_len=seq_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader