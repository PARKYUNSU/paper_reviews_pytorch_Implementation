import torch
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

def build_vocab():
    tokenizer = get_tokenizer("basic_english")
    def yield_tokens(data_iter):
        for line in data_iter:
            yield tokenizer(line)
    train_iter = PennTreebank(split="train")
    return build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])

class PTBDataset(Dataset):
    def __init__(self, split, vocab):
        self.data = list(PennTreebank(split=split))
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.data[idx])
        indices = [self.vocab[token] for token in tokens]
        return torch.tensor(indices[:-1]), torch.tensor(indices[1:])  # 입력, 타겟

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs, targets

def get_dataloaders(split="train", batch_size=32):
    vocab = build_vocab()
    dataset = PTBDataset(split=split, vocab=vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader, len(vocab)