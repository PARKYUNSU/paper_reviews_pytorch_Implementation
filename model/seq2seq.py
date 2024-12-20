import os
import zipfile
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model


import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def download_zip(url, output_path):
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"ZIP file downloaded to {output_path}")
    else:
        print(f"Failed to download. HTTP Response Code: {response.status_code}")

url = "http://www.manythings.org/anki/fra-eng.zip"
output_path = "fra-eng/fra-eng.zip"
download_zip(url, output_path)

path = os.getcwd()
zipfilename = os.path.join(path, output_path)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

lines = pd.read_csv('fra-eng/fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']


def preprocess_data(file_path, num_samples=60000):
    lines = pd.read_csv(file_path, names=['src', 'tar', 'lic'], sep='\t')
    lines = lines.loc[:, 'src':'tar'].iloc[:num_samples]
    lines.tar = lines.tar.apply(lambda x: '\t ' + x + ' \n')
    return lines

def create_vocab(lines):
    src_vocab, tar_vocab = set(), set()
    for line in lines.src:
        src_vocab.update(line)
    for line in lines.tar:
        tar_vocab.update(line)
    return sorted(list(src_vocab)) , sorted(list(tar_vocab))

def encode_data(lines, src_to_index, tar_to_index):
    encoder_input, decoder_input, decoder_target = [], [], []
    for src_line, tar_line in zip(lines.src, lines.tar):
        