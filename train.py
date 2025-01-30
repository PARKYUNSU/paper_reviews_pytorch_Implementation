import torch
import torch.optim as optim
from model.vit import Vision_Transformer
from data import cifar_10
from model.config import get_b16_config

import matplotlib.pyplot as plt
import numpy as np
import os

# 학습 함수
def train(model, train_loader, test_loader, epochs, learning_rate, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델을 CUDA로 이동
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터도 CUDA로 이동
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        evaluate(model, test_loader, device)
        
    print('Training finished.')

def visualize_attention(attentions, layer_idx=0, save_path=None):
    # 특정 레이어의 어텐션 맵을 시각화
    attention = attentions[layer_idx].detach().cpu().numpy()

    # 첫 번째 어텐션 헤드 선택
    attention_map = attention[0, 0, :, :]  # [B, num_heads, num_patches, num_patches]
    
    # 어텐션 맵을 이미지 형태로 시각화
    plt.imshow(attention_map, cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Map - Layer {layer_idx + 1}')
    
    # PNG 파일로 저장
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        attention_img_path = os.path.join(save_path, f'attention_layer_{layer_idx+1}.png')
        plt.savefig(attention_img_path)
        print(f'Attention map saved at {attention_img_path}')
    
    # 화면에 표시
    plt.show()

# 평가 함수
def evaluate(model, test_loader, device, save_path=None):
    model.eval()
    correct = 0
    total = 0
    attentions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터도 CUDA로 이동
            outputs, layer_attentions = model(inputs)  # 어텐션 맵도 함께 받음
            attentions.append(layer_attentions)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')
    
    # 마지막 배치의 첫 번째 레이어 어텐션 맵 시각화 및 저장
    visualize_attention(attentions, layer_idx=0, save_path=save_path)

# 모델 학습과 평가를 동시에 처리하는 함수
def main(pretrained_path, epochs, batch_size, learning_rate, save_path='./attention_maps'):
    # device 설정 (cuda 사용 가능하면 cuda, 아니면 cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CIFAR-10 데이터셋 로드
    train_loader, test_loader = cifar_10(batch_size)

    # Vision Transformer 모델 준비
    config = get_b16_config()  # ViT-B/16 config
    model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=True, pretrained_path=pretrained_path)
    
    # 학습 시작
    print("Starting training...")
    train(model=model,
          train_loader=train_loader,
          test_loader=test_loader,
          epochs=epochs,
          learning_rate=learning_rate,
          device=device)  # device 전달
    
    # 학습이 끝난 후 바로 평가 실행
    print("Starting evaluation...")
    evaluate(model=model, test_loader=test_loader, device=device, save_path=save_path)