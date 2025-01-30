import torch
import torch.optim as optim
from model.vit import Vision_Transformer
from data import cifar_10

def train(model, train_loader, test_loader, epochs, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        
        evaluate(model, test_loader)
        
    print('Training finished.')


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')

def main(pretrained_path, epochs, batch_size, learning_rate):
    model = Vision_Transformer(img_size=224, num_classes=10, in_channels=3, pretrained=True, pretrained_path=pretrained_path)
    
    train_loader, test_loader = cifar_10(batch_size)
    
    train(model, train_loader, test_loader, epochs, learning_rate)

if __name__ == "__main__":
    pass