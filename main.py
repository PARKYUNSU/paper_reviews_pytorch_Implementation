import torch
import argparse
import train
from model.config import get_b16_config
from data import cifar_10
from model.vit import Vision_Transformer
from utils import save_model

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script to train, evaluate, and visualize the Vision Transformer model.'
    )
    parser.add_argument('--mode', type=str, choices=['train', 'visualize'], required=True, 
                        help="Mode of operation: 'train' for training, 'visualize' for attention map visualization")
    parser.add_argument('--pretrained_path', type=str, required=True, 
                        help='Path to the pretrained model weights')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training or evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) factor')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing parameter for cross-entropy loss')
    parser.add_argument('--save_fig', action='store_true', 
                        help='Save the loss and accuracy plot as a PNG file')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the image for visualization (required in visualize mode)')
    
    return parser.parse_args()


def visualize_attention(image_path, model, device):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    tokens = model.patch_embed(image_tensor)  # (B, num_patches, hidden_size)
    B = tokens.shape[0]
    cls_tokens = model.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
    tokens = torch.cat((cls_tokens, tokens), dim=1)   # (B, num_tokens, hidden_size)
    tokens = tokens + model.pos_embed

    attn_module = model.encoder.layers[0].attn
    attn_module.vis = True

    _, attn_probs = attn_module(tokens)  # attn_probs shape: (B, num_heads, num_tokens, num_tokens)
    attn = attn_probs[0, 0].detach().cpu().numpy()  # shape: (num_tokens, num_tokens)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.title("Attention Map - First Head of First Encoder Block")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    return plt.gcf()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        train_loader, test_loader = cifar_10(batch_size=args.batch_size)
        config = get_b16_config()
        model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=False)
        model = model.to(device)
        pretrained_weights = torch.load(args.pretrained_path, map_location=device)
        model.load_from(pretrained_weights)



        print("Starting training...")
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
        train.train(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.epochs,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    save_fig=args.save_fig)
        save_model(model, "fine_tuned_model.pth")

    elif args.mode == 'visualize':
        if args.image_path is None:
            raise ValueError("Visualization mode requires --image_path argument.")
        
        print("Starting visualization...")

        config = get_b16_config()
        model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3,
                                   pretrained=True, pretrained_path=args.pretrained_path)
        model = model.to(device)
        model.encoder.layers[0].attn.vis = True
        fig = visualize_attention(args.image_path, model, device)
        fig.savefig("attention_map.png")
        print("Figure saved as attention_map.png")