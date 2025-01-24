import torch
import torch.nn as nn
from encoder import Encoder
from pathc_embedding import Patch_Embedding

class Vision_Transformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, in_channels=3):
        super(Vision_Transformer, self).__init__()
        self.num_classes = num_classes

        # patch embedding
        self.patch_embed = Patch_Embedding(config, img_size, in_channels)
        num_patches = self.patch_embed.num_patches

        # Cls
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.pos_drop = nn.Dropout(config.transformer["dropout_rate"])

        # Transformer Encoder
        self.encoder = Encoder(config)

        # Classification Head
        self.head = nn.Linear(config.hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)
    
    def _init_module(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)