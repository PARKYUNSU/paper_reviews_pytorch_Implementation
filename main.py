import torch
import argparse
import train
import os
from model.config import get_b16_config
from data import cifar_10
from model.vit import Vision_Transformer
from utils import save_model

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from transformers import ViTConfig, ViTForImageClassification
from collections import OrderedDict

PRETRAINED_MODEL_PATH = "vit_base_patch16.pth"

if not os.path.exists(PRETRAINED_MODEL_PATH):
    print("Downloading pretrained ViT model")
    pretrained_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", ignore_mismatched_sizes=True)
    torch.save(pretrained_model.state_dict(), PRETRAINED_MODEL_PATH)
    print(f"Downloaded pretrained ViT model: {PRETRAINED_MODEL_PATH}")
else:
    print(f"Model name: {PRETRAINED_MODEL_PATH}")

def convert_state_dict(state_dict):
    """ 
    HuggingFace ViT 모델의 state_dict 키를 Vision_Transformer 모델의 키와 맞게 변환하는 함수
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("embeddings.patch_embeddings.projection"):
            new_k = k.replace("embeddings.patch_embeddings.projection", "patch_embed.proj")
        elif k.startswith("embeddings.cls_token"):
            new_k = k.replace("embeddings.cls_token", "cls_token")
        elif k.startswith("embeddings.position_embeddings"):
            new_k = k.replace("embeddings.position_embeddings", "pos_embed")
        elif k.startswith("encoder.layer."):
            parts = k.split(".")
            layer_idx = parts[2]
            new_key_suffix = ".".join(parts[3:])
            new_key_suffix = new_key_suffix.replace("attention.attention.query", "attn.query_dense")
            new_key_suffix = new_key_suffix.replace("attention.attention.key", "attn.key_dense")
            new_key_suffix = new_key_suffix.replace("attention.attention.value", "attn.value_dense")
            new_key_suffix = new_key_suffix.replace("attention.output.dense", "attn.output_dense")
            new_key_suffix = new_key_suffix.replace("intermediate.dense", "mlp.fc1")
            new_key_suffix = new_key_suffix.replace("output.dense", "mlp.fc2")
            new_key_suffix = new_key_suffix.replace("layernorm_before", "norm1")
            new_key_suffix = new_key_suffix.replace("layernorm_after", "norm2")
            new_k = f"encoder.layers.{layer_idx}.{new_key_suffix}"
        
        new_state_dict[new_k] = v
    
    return new_state_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script to train, evaluate, and visualize the Vision Transformer model.'
    )
    parser.add_argument('--mode', type=str, choices=['train', 'visualize'], required=True, help="Mode of operation: 'train' for training, 'visualize' for attention map visualization")
    # parser.add_argument('--pretrained_path', type=str, required=True, help='Path to the pretrained model weights')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training or evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--save_fig', action='store_true', help='Save the loss and accuracy plot as a PNG file')
    parser.add_argument('--image_path', type=str, default=None,help='Path to the image for visualization (required in visualize mode)')

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

        # pretrained_weights = torch.load(args.pretrained_path, map_location=device)
        # converted_weights = convert_state_dict(pretrained_weights)
        # model.load_state_dict(converted_weights, strict=False)
        # print("check pre-trained model")

        print("Starting training...")
        train.train(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    device=device,
                    save_fig=args.save_fig)
        save_model(model, "fine_tuned_model.pth")

    elif args.mode == 'visualize':
        if args.image_path is None:
            raise ValueError("Visualization mode requires --image_path argument.")
        
        print("Starting visualization...")
        config = get_b16_config()
        # pretrained_weights = torch.load(args.pretrained_path, map_location=device, weights_only=True)
        model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=False)
        # model.load_state_dict(pretrained_weights, strict=False)
        model = model.to(device)
        model.encoder.layers[0].attn.vis = True
        fig = visualize_attention(args.image_path, model, device)
        fig.savefig("attention_map.png")
        print("Figure saved as attention_map.png")