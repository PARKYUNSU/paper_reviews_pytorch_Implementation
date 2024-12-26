import torch
import torch.nn as nn
import torch.optim as optim
from model.text_cl_model import TextClassificationModel
from torchtext.data.utils import get_tokenizer
from data import load_data, create_dataloaders
from train import train_one_epoch
from eval import evaluate_one_epoch
from utils import save_checkpoint, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = get_tokenizer("basic_english")
x_train, x_test, y_train, y_test, vocab = load_data(
    file_path='sarcasm.json', 
    tokenizer=tokenizer, 
    min_freq=2, 
    max_tokens=1000,
    url='https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
)

train_loader, valid_loader = create_dataloaders(
    x_train, x_test, y_train, y_test, vocab, tokenizer, 
    batch_size=32, max_sequence_length=120, device=device
)

# 모델 정의
config = {
    'num_classes': 2, 
    'vocab_size': len(vocab), 
    'embedding_dim': 16, 
    'hidden_size': 32, 
    'num_layers': 2, 
    'bidirectional': True,
}

model = TextClassificationModel(**config)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 및 평가 루프
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    print(f"Epoch {epoch}/{num_epochs}")
    
    train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = evaluate_one_epoch(model, valid_loader, loss_fn, device)
    
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, path=f"LSTM_Best.pth")