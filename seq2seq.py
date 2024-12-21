import os
import zipfile
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import tensorflow as tf
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
SRC_VOCAB, TAR_VOCAB = None, None

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
        encoder_input.append([src_to_index[char] for char in src_line])
        decoder_input.append([tar_to_index[char] for char in tar_line])
        decoder_target.append([tar_to_index[char] for char in tar_line[1:]])
    return encoder_input, decoder_input, decoder_target

def pad_sequences_data(encoder_input, decoder_input, decoder_target, max_src_len, max_tar_len):
    encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
    decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
    decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')
    return encoder_input, decoder_input, decoder_target

def seq2seq(src_vocab_size, tar_vocab_size, latent_dim=256):
    # encoder
    encoder_inputs = Input(shape=(None, src_vocab_size))
    encoder_lstm = LSTM(units=latent_dim, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # decoder
    decoder_inputs = Input(shape=(None, tar_vocab_size))
    decoder_lstm = LSTM(units=latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(tar_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # model
    model = Model([encoder_inputs, decoder_inputs] , decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model

def build_inference_models(encoder_inputs, encoder_states, decoder_lstm, decoder_dense, latent_dim=256):
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_inputs = Input(shape=(None, TAR_VOCAB))

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, tar_vocab_size, tar_to_index, index_to_tar, max_tar_len):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, tar_to_index['\t']] = 1.

    decoded_sentence = ""
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or len(decoded_sentence) > max_tar_len):
            stop_condition = True

        target_seq = np.zeros((1, 1, tar_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.
        states_value = [h, c]
    return decoded_sentence

def main():

    global SRC_VOCAB, TAR_VOCAB
    SRC_VOCAB, TAR_VOCAB = create_vocab(lines)
    src_to_index = {char: i+1 for i, char in enumerate(SRC_VOCAB)}
    tar_to_index = {char: i+1 for i, char in enumerate(TAR_VOCAB)}
    index_to_tar = {i+1: char for i, char in enumerate(TAR_VOCAB)}

    encoder_input, decoder_input, decoder_target = encode_data(lines, src_to_index, tar_to_index)
    max_src_len = max([len(line) for line in lines.src])
    max_tar_len = max([len(line) for line in lines.tar])
    encoder_input, decoder_input, decoder_target = pad_sequences_data(encoder_input, decoder_input, decoder_target, max_src_len, max_tar_len)
    encoder_input = to_categorical(encoder_input)
    decoder_input = to_categorical(decoder_input)
    decoder_target = to_categorical(decoder_target)

    model = seq2seq(len(SRC_VOCAB)+1, len(TAR_VOCAB)+1)
    model.fit([encoder_input, decoder_input], decoder_target, batch_size=64, epochs=40, validation_split=0.2)

    encoder_model, decoder_model = build_inference_models(model.input[0], model.layers[2].output, model.layers[3], model.layers[4])

    for seq_index in [3, 50, 100, 300, 1001]:
        input_seq = encoder_input[seq_index:seq_index+1]
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, len(TAR_VOCAB)+1, tar_to_index, index_to_tar, max_tar_len)
        print(f"입력 문장: {lines.src.iloc[seq_index]}")
        print(f"정답 문장: {lines.tar.iloc[seq_index][2:-2]}")
        print(f"번역 문장: {decoded_sentence.strip()}")

if __name__ == "__main__":
    main()