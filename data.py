import numpy as np
import random

def generate_random_data(n):
    SOS_token = np.array([2])
    EOS_token = np.array([3])
    length = 8
    
    data = []
    
    # 1,1,1,1,1 -> 1,1,1,1,1
    for i in range(n // 3):
        X = np.concatenate((SOS_token, np.ones(length), EOS_token))
        y = np.concatenate((SOS_token, np.ones(length), EOS_token))
        data.append([X, y])

    # 0,0,0,0 -> 0,0,0,0
    for i in range(n // 3):
        X = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        y = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        data.append([X, y])

    # 1,0,1,0 -> 1,0,1,0,1
    for i in range(n // 3):
        X = np.zeros(length)
        start = random.randint(0, 1)
        
        X[start::2] = 1
        
        y = np.zeros(length)
        if X[-1] == 0:
            y[::2] = 1
        else:
            y[1::2] = 1
        
        X = np.concatenate((SOS_token, X, EOS_token))
        y = np.concatenate((SOS_token, y, EOS_token))
        data.append([X, y])

    np.random.shuffle(data)
    
    return data

def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
    batches = []
    for idx in range(0, len(data), batch_size):
        if idx + batch_size < len(data):
            if padding:
                max_batch_length = 0
                for seq in data[idx : idx + batch_size]:
                    if len(seq[0]) > max_batch_length:
                        max_batch_length = len(seq[0])

                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx][0])
                    data[idx + seq_idx][0] += [padding_token] * remaining_length
                    data[idx + seq_idx][1] += [padding_token] * remaining_length
            
            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")
    
    return batches