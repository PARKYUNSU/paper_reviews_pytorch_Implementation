# import argparse
# import train
# import eval
# from model.config import get_b16_config

# def parse_args():
#     parser = argparse.ArgumentParser(description='Main script to either train or evaluate the model.')
#     parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, 
#                         help="Mode of operation: 'train' for training and 'eval' for evaluation")
#     parser.add_argument('--pretrained_path', type=str, help='Path to the pretrained model weights')
#     parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
#     parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training or evaluation')
#     parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    
#     return parser.parse_args()

# if __name__ == "__main__":
#     args = parse_args()

#     if args.mode == 'train':
#         print("Starting training...")
#         train.train(model=None,
#                     train_loader=None,
#                     test_loader=None,
#                     epochs=args.epochs,
#                     learning_rate=args.learning_rate)
    
#     elif args.mode == 'eval':
#         print("Starting evaluation...")
#         eval.evaluate(pretrained_path=args.pretrained_path, batch_size=args.batch_size)

import torch
import argparse
import train
import eval
from model.vit import Vision_Transformer
from model.config import get_b16_config

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

    if args.mode == 'train':
        print("Starting training...")

        # Vision Transformer 모델 초기화
        config = get_b16_config()  # ViT-B/16 config
        model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=True, pretrained_path=args.pretrained_path)
        
        # 데이터 로더 준비
        # 실제 cifar_10 데이터 로더를 사용하는 것으로 수정
        train_loader, test_loader = None, None  # 데이터 로딩 로직 필요
        
        # 모델 훈련
        train.train(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate)
    
    elif args.mode == 'eval':
        print("Starting evaluation...")

        # Vision Transformer 모델 초기화 (pretrained=False, but weights loaded)
        config = get_b16_config()  # ViT-B/16 config
        model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=False)
        model.load_state_dict(torch.load(args.pretrained_path))  # 다운받은 가중치 로드
        
        # 평가 함수 실행
        eval.evaluate(model=model, batch_size=args.batch_size)