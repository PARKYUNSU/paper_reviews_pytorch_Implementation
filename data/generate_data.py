import os
import urllib.request
import tarfile
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class IMDBDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data = []
        self.labels = []
        self.tokenizer = lambda x: re.findall(r"\b\w+\b", x.lower())  # 간단한 토크나이저

        # Load data
        data_path = os.path.join(data_dir, split)
        for label in ["pos", "neg"]:
            dir_path = os.path.join(data_path, label)
            for filename in os.listdir(dir_path):
                with open(os.path.join(dir_path, filename), "r", encoding="utf-8") as file:
                    self.data.append(self.tokenizer(file.read()))
                    self.labels.append(1 if label == "pos" else 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts, labels

def download_imdb(data_dir):
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    file_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    extract_path = os.path.join(data_dir, "aclImdb")

    if not os.path.exists(extract_path):
        print("Downloading IMDB dataset...")
        urllib.request.urlretrieve(url, file_path)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Dataset extracted.")
    else:
        print("IMDB dataset already downloaded.")

def get_dataloaders(data_dir="data", batch_size=32):
    download_imdb(data_dir)
    train_dataset = IMDBDataset(os.path.join(data_dir, "aclImdb"), split="train")
    test_dataset = IMDBDataset(os.path.join(data_dir, "aclImdb"), split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, test_loader, len(train_dataset.data)