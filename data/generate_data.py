import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords
from collections import Counter
import re
import string


# IMDB 데이터셋 전처리 및 Dataset 클래스 정의
class IMDBDataset(Dataset):
    def __init__(self, data, labels, vocab, seq_length):
        """
        IMDB 데이터셋 초기화

        Args:
            data (list of str): 리뷰 텍스트 리스트
            labels (list of int): 리뷰 라벨 리스트 (0: negative, 1: positive)
            vocab (dict): 단어를 인덱스로 매핑한 사전
            seq_length (int): 고정된 시퀀스 길이
        """
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = self.data[idx]
        label = self.labels[idx]

        # 리뷰를 단어 인덱스로 변환
        tokenized_review = [self.vocab.get(word, 0) for word in preprocess_string(review).split()]
        
        # 고정된 길이로 패딩
        padded_review = pad_sequence(tokenized_review, self.seq_length)
        
        return torch.tensor(padded_review, dtype=torch.long), torch.tensor(label, dtype=torch.float32)


def preprocess_string(s):
    """리뷰 전처리"""
    # 특수 문자 제거
    s = re.sub(r"[^\w\s]", '', s)
    # 공백 제거
    s = re.sub(r"\s+", ' ', s)
    # 숫자 제거
    s = re.sub(r"\d", '', s)
    # 소문자로 변환
    return s.strip().lower()  # 공백 제거 후 소문자로 변환


def build_vocab(reviews, max_vocab_size=1000):
    """
    단어 사전 생성

    Args:
        reviews (list of str): 리뷰 텍스트 리스트
        max_vocab_size (int): 최대 단어 수

    Returns:
        dict: 단어를 인덱스로 매핑한 사전
    """
    word_list = []
    stop_words = set(stopwords.words('english'))
    for review in reviews:
        for word in preprocess_string(review).split():
            if word not in stop_words:  # 불용어 제거
                word_list.append(word)
    
    corpus = Counter(word_list)
    most_common = corpus.most_common(max_vocab_size)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}  # 0은 패딩에 사용
    return vocab


def pad_sequence(sequence, seq_length):
    """
    고정된 길이로 패딩

    Args:
        sequence (list of int): 단어 인덱스 리스트
        seq_length (int): 고정된 시퀀스 길이

    Returns:
        list of int: 패딩된 시퀀스
    """
    if len(sequence) >= seq_length:
        return sequence[:seq_length]
    else:
        return [0] * (seq_length - len(sequence)) + sequence


def prepare_imdb_data(csv_path, seq_length, max_vocab_size=1000):
    """
    IMDB 데이터셋 준비

    Args:
        csv_path (str): IMDB 데이터셋 경로
        seq_length (int): 고정된 시퀀스 길이
        max_vocab_size (int): 최대 단어 수

    Returns:
        Dataset, Dataset, dict: 훈련 데이터셋, 검증 데이터셋, 단어 사전
    """
    # CSV 파일 로드
    df = pd.read_csv(csv_path)

    # Null 값 제거
    df = df.dropna(subset=['review', 'sentiment'])

    # 리뷰와 라벨 추출
    reviews = df['review'].values
    labels = [1 if sentiment.lower() == 'positive' else 0 for sentiment in df['sentiment'].values]
    
    # 데이터셋 분할
    train_size = int(0.8 * len(reviews))
    train_reviews, val_reviews = reviews[:train_size], reviews[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]

    # 단어 사전 생성
    vocab = build_vocab(train_reviews, max_vocab_size)
    
    # 데이터셋 생성
    train_dataset = IMDBDataset(train_reviews, train_labels, vocab, seq_length)
    val_dataset = IMDBDataset(val_reviews, val_labels, vocab, seq_length)
    
    return train_dataset, val_dataset, vocab