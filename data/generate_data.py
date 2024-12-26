import pandas as pd
import torch
import urllib
import os
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence


class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        super().__init__()
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        return self.vocab(self.tokenizer(text)), label


def load_data(file_path, tokenizer, min_freq=2, max_tokens=1000, url=None):
    # 데이터 다운로드 (필요 시)
    if url and not os.path.exists(file_path):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, file_path)

    # 데이터 로드
    with open(file_path, 'r') as f:
        data = pd.DataFrame(json.load(f))

    # 단어 사전 구축
    vocab = build_vocab_from_iterator(
        (tokenizer(text) for text in data['headline']), 
        specials=['<UNK>'], 
        min_freq=min_freq, 
        max_tokens=max_tokens
    )
    vocab.set_default_index(vocab['<UNK>'])

    # Train/Test 분리

    x_train, x_test, y_train, y_test = train_test_split(
        data['headline'], 
        data['is_sarcastic'], 
        stratify=data['is_sarcastic'], 
        test_size=0.2, 
        random_state=123
    )

    return x_train, x_test, y_train, y_test, vocab


def collate_batch(batch, max_sequence_length, device):
    text_list, label_list = [], []
    for text, label in batch:
        text_list.append(torch.tensor(text[:max_sequence_length], dtype=torch.int64))
        label_list.append(label)
    
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return text_list.to(device), label_list.to(device)


def create_dataloaders(x_train, x_test, y_train, y_test, vocab, tokenizer, batch_size, max_sequence_length, device):
    train_ds = SarcasmDataset(x_train, y_train, vocab, tokenizer)
    valid_ds = SarcasmDataset(x_test, y_test, vocab, tokenizer)

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: collate_batch(x, max_sequence_length, device)
    )
    
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_batch(x, max_sequence_length, device)
    )

    return train_loader, valid_loader