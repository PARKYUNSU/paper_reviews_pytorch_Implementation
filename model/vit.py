import numpy as np
import torch
import torch.nn as nn
import ndimage

from .encoder import Encoder
from .pathc_embedding import Patch_Embedding
from .utils import np2th

class Vision_Transformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, in_channels=3,  pretrained=False, pretrained_path=None):
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

        if pretrained and pretrained_path is not None:
            self.load_from(torch.load(pretrained_path, map_location="cpu"))
            
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

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, num_patches, hidden_size)

        # Cls
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, num_patches+1, hidden_size)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer Encoder
        x = self.encoder(x)

        # Classification
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits
    

    def load_from(self, weights):
        # Classification head load
        self.head.weight.data.copy_(np2th(weights["head/kernel"]).t())
        self.head.bias.data.copy_(np2th(weights["head/bias"]).t())
        
        # Patch embedding weights load (conv인 경우)
        self.patch_embed.proj.weight.data.copy_(np2th(weights["embedding/kernel"], conv=True))
        self.patch_embed.proj.bias.data.copy_(np2th(weights["embedding/bias"]))
        
        # Cls load
        self.cls_token.data.copy_(np2th(weights["cls"]))
        
        # Positional embedding load
        posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
        if posemb.size() == self.pos_embed.size():
            self.pos_embed.data.copy_(posemb)
        else:
            # 크기가 다르면 2D 보간(ndimage.zoom) 사용
            
            ntok_new = self.pos_embed.size(1)
            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
            gs_old = int(np.sqrt(len(posemb_grid)))
            gs_new = int(np.sqrt(ntok_new - 1))
            posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
            zoom = (gs_new / gs_old, gs_new / gs_old, 1)
            posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
            posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
            new_posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
            self.pos_embed.data.copy_(np2th(new_posemb))