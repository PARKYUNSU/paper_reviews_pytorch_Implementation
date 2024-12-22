import os
import urllib.request
import tarfile
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class IMDBDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.data = []
        self.labels = []
        self.tokenizer = lambda x: re.findall(r"\b\w+\b", x.lower())  # 간단한 토크나이저
        self.vocab = {}

        # Load data
        data_path = os.path.join(data_dir, "aclImdb", mode)
        for label in ["pos", "neg"]:
            dir_path = os.path.join(data_path, label)
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory {dir_path} not found.")
            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)
                if not os.path.isfile(filepath):
                    continue
                with open(filepath, "r", encoding="utf-8") as file:
                    tokens = self.tokenizer(file.read())
                    if len(tokens) > 0:  # 빈 시퀀스 제외
                        self.data.append(tokens)
                        self.labels.append(1 if label == "pos" else 0)

        # Build vocab
        self.build_vocab()

    def build_vocab(self):
        word_freq = {}
        for tokens in self.data:
            for token in tokens:
                word_freq[token] = word_freq.get(token, 0) + 1

        sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocab = {word: idx + 2 for idx, (word, _) in enumerate(sorted_vocab)}  # Start at 2 to reserve 0 and 1
        self.vocab["<pad>"] = 0
        self.vocab["<unk>"] = 1

    def encode(self, tokens):
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.encode(self.data[idx])
        label = self.labels[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # (batch_size, seq_len)
    targets = torch.tensor(targets)  # (batch_size)
    return inputs, targets


def download_imdb(data_dir):
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    file_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    extract_path = os.path.join(data_dir, "aclImdb")

    if not os.path.exists(extract_path):
        print("Downloading IMDB dataset...")
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, file_path)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Dataset extracted.")
    else:
        print("IMDB dataset already downloaded.")


def get_dataloaders(data_dir="data", batch_size=32):
    # Download IMDB dataset if not already present
    download_imdb(data_dir)

    # Create train and test datasets
    train_dataset = IMDBDataset(data_dir, mode="train")
    test_dataset = IMDBDataset(data_dir, mode="test")

    # Create DataLoader for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Return DataLoaders and vocab size
    return train_loader, test_loader, len(train_dataset.vocab)