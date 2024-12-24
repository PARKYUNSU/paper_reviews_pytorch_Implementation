import yfinance as yf
import pandas as pd
import numpy as np


def download_stock_data(ticker, start_date, end_date):
    """특정 주식 데이터를 Yahoo Finance에서 다운로드"""
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def save_stock_data(data, filename):
    """주식 데이터를 CSV 파일로 저장"""
    data.to_csv(filename)
    print(f"Stock data saved to {filename}")


def prepare_stock_data(file_path, seq_length):
    """CSV 파일에서 데이터를 불러와 시계열 데이터로 변환"""
    stock_data = pd.read_csv(file_path, index_col=0)
    # 'Close' 열을 기준으로 시계열 생성
    prices = stock_data['Close'].values
    sequences = []
    for i in range(len(prices) - seq_length):
        seq = prices[i:i + seq_length]
        label = prices[i + seq_length]
        sequences.append((seq, label))
    return sequences
