import torch

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    print(f"Model loaded from {load_path}")