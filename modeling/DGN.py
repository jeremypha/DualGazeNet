# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import math

from typing import List, Tuple
from einops import rearrange

from .transformer import SpatialTransformer, BasicTransformerBlock
from .common import LayerNorm2d, MLP
from .backbones.build_backbone import build_backbone

from config import Config


class DGN(nn.Module):
    """
    Args:
        backbone (str): Backbone type, e.g., 'B', 'L', 'S'
        pretrained (bool): Whether to use pretrained weights
    """
    def __init__(
        self,
        backbone='L',
        pretrained=True,
    ) -> None:
        
        super().__init__()
        self.config = Config()

        backbone_name = self.config.backbone[backbone]

        self.encoder = build_backbone(backbone_name, pretrained)

        # Freeze Hiera backbone and add adapters for parameter-efficient tuning
        if 'hiera' in backbone_name:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            blocks = []
            for block in self.encoder.blocks:
                blocks.append(
                    Adapter(block)  # Add learnable adapter to each block
                )
            self.encoder.blocks = nn.Sequential(*blocks)

        self.mask_decoder = MaskDecoder(transformer_dims=self.config.dec_dims[backbone_name])
        
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.tensor:
        """
        Args:
            images (torch.Tensor): Input images of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Predicted mask of shape [B, 1, H, W]
        """

        feas = list(self.encoder(images))

        # Reshape Swin Transformer features from sequence to spatial format
        if 'swin' in self.config.backbone:
            feas = [fea.reshape(
                fea.shape[0], 
                int(math.sqrt(fea.shape[1])), 
                int(math.sqrt(fea.shape[1])), -1
                ).permute(0, 3, 1, 2) 
            for fea in feas]

        mask = self.mask_decoder(features_list=feas)

        return mask


# Adapted from https://github.com/WZH0120/SAM2-UNet/blob/main/SAM2UNet.py
class Adapter(nn.Module):
    """
    Adapter module for parameter-efficient fine-tuning of frozen backbone.
    
    Adds a small learnable network to each transformer block while keeping 
    the original block frozen.
    """
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk  # Original frozen transformer block
        dim = blk.attn.qkv.in_features
        
        # Small learnable adapter network (bottleneck architecture)
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32), 
            nn.GELU(),       
            nn.Linear(32, dim),  
            nn.GELU()
        )

    def forward(self, x):

        prompt = self.prompt_learn(x)
        prompted = x + prompt
        net = self.block(prompted)

        return net


class MaskDecoder(nn.Module):
    """
    Uses both forward and reverse transformer paths with cross-scale feature fusion
    for precise mask prediction.
    """
    def __init__(
        self,
        transformer_dims: List[int],
    ) -> None:
        
        super().__init__()

        # Learnable fixed token for query initialization
        self.fix_token = nn.Embedding(1, transformer_dims[-1])
        self.num_dim = len(transformer_dims)

        # Initial transformers for forward path
        self.init_transformers = nn.ModuleList()
        for i in range(self.num_dim):
            self.init_transformers.append(BasicTransformerBlock(
                dim=transformer_dims[i],
                n_heads=8,
                d_head=32,
                context_dim=transformer_dims[i],  # Self-attention with skip
                skip_self=True,  # Skip self-attention, only cross-attention
            )
        )

        # MLPs for dimension projection between scales
        self.init_mlps = nn.ModuleList()
        for i in range(self.num_dim - 1):
            self.init_mlps.append(MLP(
                transformer_dims[i + 1],
                transformer_dims[i + 1],
                transformer_dims[i],   
                1 
            )
        )

        # Reverse transformers for backward path
        self.reverse_transformers = nn.ModuleList()
        for i in range(self.num_dim):
            self.reverse_transformers.append(SpatialTransformer(
                in_channels=transformer_dims[i],
                n_heads=8,
                d_head=32,
                depth=1,
                context_dim=transformer_dims[i],  # Use tokens as context
                skip_self=True,
            )
        )

        # Upsampling layers for feature fusion
        self.fuse_upscalings = nn.ModuleList()    
        for i in range(self.num_dim - 1):
            self.fuse_upscalings.append(nn.Sequential(
                nn.ConvTranspose2d(transformer_dims[i + 1], transformer_dims[i], kernel_size=2, stride=2),
                LayerNorm2d(transformer_dims[i]),
                nn.GELU(),
            )
        )

        # Final output projection network
        self.output_hypernetworks_mlp = MLP(
            transformer_dims[0],           # Input: finest scale token
            transformer_dims[0],           # Hidden dimension  
            transformer_dims[0] // 8,      # Output: reduced dimension
            3                              # Number of layers
        )

        # Final upsampling to original resolution
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dims[0], transformer_dims[0] // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dims[0] // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dims[0] // 4, transformer_dims[0] // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

    def forward(
        self,
        features_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features_list (List[torch.Tensor]): Multi-scale features from encoder
            where each has shape [B, C, H, W]
                
        Returns:
            torch.Tensor: Predicted mask of shape [B, 1, H, W]
        """
        
        b, c, h, w = features_list[-1].shape

        # Initialize learnable query token
        fix_token = self.fix_token.weight.expand(b, -1, -1)  # [B, 1, C]
        token = fix_token    # [B, 1, C]

        tokens = [token]  # Store tokens at different scales
        
        # Forward path: coarse to fine processing
        for i in reversed(range(self.num_dim)):
            # Get feature at current scale and reshape to sequence
            fea = features_list[i]
            fea = rearrange(fea, 'b c h w -> b (h w) c')
            
            # Cross-attention between token and features
            token = self.init_transformers[i](token, fea)   
            
            # Project to finer scale if not at finest level
            if i != 0:
                token = self.init_mlps[i - 1](token)
                tokens.insert(0, token)  # Store token for reverse path

        # Reverse path: fine to coarse processing with feature fusion
        src = features_list[-1]  # Start from coarsest features
        for i in reversed(range(self.num_dim)):
            # Spatial transformer with token guidance
            src = self.reverse_transformers[i](src, tokens[i])
            
            # Upsample and fuse with features from finer scale
            if i != 0:
                upscaled_src = self.fuse_upscalings[i - 1](src)
                src = upscaled_src + features_list[i - 1]  # Residual fusion

        # Final mask generation
        upscaled_embedding = self.output_upscaling(src)              # [B, C, H, W]
        hyper_in = self.output_hypernetworks_mlp(token)              # [B, 1, C]
        b, c, h, w = upscaled_embedding.shape
        mask = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # [B, 1, H, W]

        return mask