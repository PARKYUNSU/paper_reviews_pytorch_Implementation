import torch
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Vocab 생성 함수
def build_vocab():
    tokenizer = get_tokenizer("basic_english")

    # 토큰 생성기 함수
    def yield_tokens(data_iter):
        for line in data_iter:
            yield tokenizer(line)

    # PennTreebank의 train 데이터에서 vocab 생성
    train_iter = PennTreebank(split="train")
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter), specials=["<unk>", "<pad>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

# PTB Dataset 클래스
class PTBDataset(Dataset):
    def __init__(self, split, vocab):
        self.data = list(PennTreebank(split=split))  # train/valid/test 데이터 로드
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.data[idx])  # 토큰화
        indices = [self.vocab[token] for token in tokens]  # 인덱스 변환
        return torch.tensor(indices[:-1]), torch.tensor(indices[1:])  # 입력, 타겟

# 배치 처리를 위한 Collate 함수
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # 입력 패딩
    targets = pad_sequence(targets, batch_first=True, padding_value=0)  # 타겟 패딩
    return inputs, targets

# DataLoader 생성 함수
def get_dataloaders(split="train", batch_size=32):
    vocab = build_vocab()  # Vocab 생성
    dataset = PTBDataset(split=split, vocab=vocab)  # Dataset 생성
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)  # DataLoader 생성
    return dataloader, len(vocab)

# 실행 테스트
if __name__ == "__main__":
    train_loader, vocab_size = get_dataloaders(split="train", batch_size=32)
    print("Vocab size:", vocab_size)

    # 데이터 로드 테스트
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print("Inputs shape:", inputs.shape)
        print("Targets shape:", targets.shape)
        if batch_idx == 2:  # 예시로 3개 배치만 출력
            break