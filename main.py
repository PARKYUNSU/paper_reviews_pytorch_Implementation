import argparse
from train import train
from eval import evaluate

def main():
    parser = argparse.ArgumentParser(description="LSTM Training and Evaluation")
    parser.add_argument('--mode', type=str, default="train", choices=["train", "eval"],
                        help="Select mode: 'train' or 'eval'")
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "eval":
        preds = evaluate()
        print("Sample Predictions:", preds[:10])

if __name__ == "__main__":
    main()