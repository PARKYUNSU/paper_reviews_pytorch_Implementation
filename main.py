import argparse
import train
import eval
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
        train.train(model=None,
                    train_loader=None,
                    test_loader=None,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate)
    
    elif args.mode == 'eval':
        print("Starting evaluation...")
        eval.evaluate(pretrained_path=args.pretrained_path, batch_size=args.batch_size)