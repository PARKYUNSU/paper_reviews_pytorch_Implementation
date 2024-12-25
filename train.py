import torch
import torch.nn as nn
import numpy as np

def train_model(train_loader, valid_loader, model, criterion, optimizer, num_epochs, device, clip=5):
    valid_loss_min = np.Inf  # Validation 손실 초기화
    epoch_tr_loss, epoch_vl_loss = [], []
    epoch_tr_acc, epoch_vl_acc = [], []

    for epoch in range(num_epochs):
        train_losses = []
        train_acc = 0.0
        model.train()  # 모델을 학습 모드로 설정

        # 학습 루프
# 학습 루프
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # LSTM hidden state 초기화
            h = model.init_hidden(inputs.size(0), device)  # batch_size와 device 전달
            h = tuple([each.data for each in h])  # hidden state와 cell state 분리

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # Forward
            output, h = model(inputs, h)

            # 손실 계산 및 역전파
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()

            # Gradient Clipping
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # 손실 및 정확도 계산
            train_losses.append(loss.item())
            train_acc += acc(output, labels)

        # 검증 루프
        val_losses = []
        val_acc = 0.0
        model.eval()  # 모델을 평가 모드로 설정
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                val_h = model.init_hidden(inputs.size(0))
                val_h = tuple([each.data for each in val_h])

                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())
                val_losses.append(val_loss.item())
                val_acc += acc(output, labels)

        # 에포크 손실 및 정확도 저장
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc / len(train_loader.dataset)
        epoch_val_acc = val_acc / len(valid_loader.dataset)

        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"train_loss: {epoch_train_loss:.6f}, val_loss: {epoch_val_loss:.6f}")
        print(f"train_accuracy: {epoch_train_acc*100:.2f}%, val_accuracy: {epoch_val_acc*100:.2f}%")

        # Validation loss가 감소하면 모델 저장
        if epoch_val_loss <= valid_loss_min:
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Validation loss decreased ({valid_loss_min:.6f} --> {epoch_val_loss:.6f}). Saving model...")
            valid_loss_min = epoch_val_loss

        print("=" * 50)

    return epoch_tr_loss, epoch_vl_loss, epoch_tr_acc, epoch_vl_acc


# 정확도 계산 함수
def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()