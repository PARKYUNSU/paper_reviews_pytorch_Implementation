import torch
import argparse
import train
from model.config import get_b16_config
from data import cifar_10
from model.vit import Vision_Transformer
from utils import save_model

import numpy as np
import cv2
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        tokens = model.patch_embed(image_tensor)  # (B, num_patches, hidden_size)
        B = tokens.shape[0]
        cls_tokens = model.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
        tokens = torch.cat((cls_tokens, tokens), dim=1)  # (B, num_tokens, hidden_size)
        tokens = tokens + model.pos_embed

        attn_module = model.encoder.layers[0].attn
        attn_module.vis = True
        _, attn_probs = attn_module(tokens)  # (B, num_heads, num_tokens, num_tokens)
    
    
    cls_attn = attn_probs[0, 0, 0, 1:].detach().cpu().numpy()  # (num_patches,)
    
    num_patches = int(np.sqrt(len(cls_attn)))  # num_patches = 14 (for 224x224 with 16x16 patches)
    attn_map = cls_attn.reshape(num_patches, num_patches)  # (14, 14)
    attn_map = cv2.resize(attn_map, (224, 224))  # (224, 224)

    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())  # Normalize (0~1)
    attn_map = 1 - attn_map
    attn_map = np.expand_dims(attn_map, axis=-1)

    original_image = np.array(original_image.resize((224, 224))) / 255.0  # Normalize 원본 이미지 (0~1)
    darkened_image = (original_image * attn_map).clip(0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(darkened_image)
    ax[1].set_title("ViT Attention Overlay")
    ax[1].axis("off")

    return fig

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
                    learning_rate=args.learning_rate,
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
        model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=True, pretrained_path=args.pretrained_path)
        model = model.to(device)
        model.encoder.layers[0].attn.vis = True
        fig = visualize_attention(args.image_path, model, device)
        fig.savefig("attention_map.png")
        print("Figure saved as attention_map.png")