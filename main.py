import torch
from data_generate import load_nsmc_data
from utils import LSTM, get_optimizer_and_criterion
from eval import evaluate

def simple_tokenizer(text):
    return text.split()

def main():
    # 하이퍼파라미터 설정
    max_length = 50
    batch_size = 64
    input_dim = 5000  # vocab 크기
    hidden_dim = 128
    layer_dim = 2
    output_dim = 1
    dropout_prob = 0.2
    num_epochs = 10
    learning_rate = 0.001

    # 데이터 로드
    train_loader, test_loader, vocab = load_nsmc_data(simple_tokenizer, max_length, batch_size)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(len(vocab), hidden_dim, layer_dim, output_dim, dropout_prob).to(device)
    optimizer, criterion = get_optimizer_and_criterion(model, lr=learning_rate)

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            hidden = model.init_hidden(inputs.size(0), device)

            optimizer.zero_grad()
            outputs = model(inputs, hidden).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()