import os
import requests

import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from itertools import chain

# Step 1: Download WikiText2 dataset


def download_wikitext2(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    file_path = os.path.join(data_dir, "wikitext-2-v1.zip")

    # 파일 다운로드
    if not os.path.exists(file_path):
        print("Downloading WikiText2 dataset...")
        response = requests.get(url, stream=True)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

    # 파일 추출
    extract_dir = os.path.join(data_dir, "wikitext-2")
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Dataset extracted to {extract_dir}")
    else:
        print(f"Dataset already extracted at {extract_dir}")

# Step 2: Tokenization and Vocabulary Building
def build_vocab(data_path, min_freq=2):
    with open(data_path, "r") as f:
        text = f.readlines()
    tokens = [word for line in text for word in line.strip().split()]
    counter = Counter(tokens)
    vocab = {word: idx + 2 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab["<unk>"] = 0
    vocab["<pad>"] = 1
    return vocab

# Step 3: Custom Dataset for WikiText2
class WikiText2Dataset(Dataset):
    def __init__(self, data_path, vocab, seq_len=50):
        self.vocab = vocab
        self.seq_len = seq_len
        with open(data_path, "r") as f:
            text = f.readlines()
        tokens = [self.vocab.get(word, self.vocab["<unk>"]) for line in text for word in line.strip().split()]
        self.data = [torch.tensor(tokens[i:i + seq_len + 1]) for i in range(0, len(tokens) - seq_len, seq_len)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # Input and target

# Step 4: Collate function for DataLoader
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=1)  # Padding index
    targets = pad_sequence(targets, batch_first=True, padding_value=1)
    return inputs, targets

# Step 5: Get DataLoader
def get_dataloaders(data_dir="data", batch_size=32, seq_len=50):
    download_wikitext2(data_dir)
    train_path = os.path.join(data_dir, "wikitext-2", "wiki.train.tokens")
    vocab = build_vocab(train_path)
    dataset = WikiText2Dataset(train_path, vocab, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return dataloader, len(vocab)