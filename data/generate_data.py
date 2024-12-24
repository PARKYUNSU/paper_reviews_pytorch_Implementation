import yfinance as yf
import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import MinMaxScaler

class SineWaveDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def download_stock_data(ticker, start_date, end_date):
    """특정 주식 데이터를 Yahoo Finance에서 다운로드"""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def save_stock_data(data, filename):
    """주식 데이터를 CSV 파일로 저장"""
    data.to_csv(filename)
    print(f"Stock data saved to {filename}")


def prepare_stock_data(file_path, seq_length, noise_factor=0.02):
    """CSV 파일에서 데이터를 불러오고 노이즈를 추가한 후 시계열 데이터로 변환"""
    # 데이터 로드
    stock_data = pd.read_csv(file_path, index_col=0)
    
    # 'Close' 열을 기준으로 시계열 데이터 생성
    prices = stock_data['Close'].values
    
    # 노이즈 추가
    noisy_prices = prices + noise_factor * np.random.normal(size=prices.shape)
    
    # 시계열 데이터 생성
    sequences = []
    for i in range(len(noisy_prices) - seq_length):
        seq = noisy_prices[i:i + seq_length]
        label = noisy_prices[i + seq_length]
        sequences.append((seq, label))
    
    return sequences