import numpy as np
import torch

def generate_sine_data(seq_length=50, num_samples=1000):
    x = np.linspace(0, num_samples * 2 * np.pi / seq_length, num_samples)
    sine_wave = np.sin(x)
    
    X, y = [], []
    for i in range(num_samples - seq_length):
        X.append(sine_wave[i:i + seq_length])
        y.append(sine_wave[i + seq_length])
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

if __name__ == "__main__":
    X, y = generate_sine_data()
    print("Generated Data Shapes:", X.shape, y.shape)
