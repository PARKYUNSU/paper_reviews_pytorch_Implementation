import torch
from torch.utils.data import Dataset, DataLoader
from Korpora import Korpora

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text)
        tokens = tokens[:self.max_length]
        input_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        padding = [0] * (self.max_length - len(input_ids))
        input_ids += padding
        return torch.tensor(input_ids), torch.tensor(label)

def load_nsmc_data(tokenizer, max_length, batch_size):
    corpus = Korpora.load("nsmc")
    train_texts, train_labels = corpus.train.texts, corpus.train.labels
    test_texts, test_labels = corpus.test.texts, corpus.test.labels

    # Vocab 생성
    all_tokens = [token for text in train_texts for token in tokenizer(text)]
    vocab = {token: idx for idx, token in enumerate(set(all_tokens), 1)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = len(vocab)

    # 데이터셋 및 데이터로더
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, vocab, max_length)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer, vocab, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab