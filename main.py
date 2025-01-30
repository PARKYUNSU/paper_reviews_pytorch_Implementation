import torch
import argparse
import train
import eval
from model.config import get_b16_config
from data import cifar_10
from model.vit import Vision_Transformer

def parse_args():
    parser = argparse.ArgumentParser(description='Main script to either train or evaluate the model.')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, 
                        help="Mode of operation: 'train' for training and 'eval' for evaluation")
    parser.add_argument('--pretrained_path', type=str, help='Path to the pretrained model weights')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training or evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # CIFAR-10 데이터로더 준비
    train_loader, test_loader = cifar_10(batch_size=args.batch_size)

    # device 설정 (cuda 사용 가능하면 cuda, 아니면 cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vision Transformer 모델 준비
    config = get_b16_config()  # ViT-B/16 config
    model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=True, pretrained_path=args.pretrained_path)

    if args.mode == 'train':
        print("Starting training...")
        train.train(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    device=device)  # device 전달
    
    elif args.mode == 'eval':
        print("Starting evaluation...")
        eval.evaluate(pretrained_path=args.pretrained_path, 
                      batch_size=args.batch_size, 
                      device=device)  # device 전달