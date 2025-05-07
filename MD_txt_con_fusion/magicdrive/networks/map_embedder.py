from typing import Any, Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.controlnet import zero_module
from einops import repeat

class BEVControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int = 320,
        conditioning_size: Tuple[int, int, int] = (25, 200, 200),  # only use 25
        block_out_channels: Tuple[int] = (32, 64, 128, 256),
    ):
        super().__init__()
        # input size   25, 200, 200
        # output size 320,  28,  50

        self.conv_in = nn.Conv2d(
            conditioning_size[0],
            block_out_channels[0],
            kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 2):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=(2, 1),
                    stride=2))
        channel_in = block_out_channels[-2]
        channel_out = block_out_channels[-1]
        self.blocks.append(
            nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=(2, 1))
        )
        self.blocks.append(
            nn.Conv2d(
                channel_in, channel_out, kernel_size=3, padding=(2, 1),
                stride=(2, 1)))

        self.conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        conditioning = repeat(conditioning, 'b ... -> (b repeat) ...', repeat=6)
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


# original rgb image embedder
class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        conditioning_size = None,  # place holder
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        # view spliting: bs,c,h,pano_w -> bs*cam,c,h,w
        x=conditioning
        per_w=x.shape[-1]//6
        x=torch.stack([x[...,:per_w],
                       x[...,per_w:per_w*2],
                       x[...,per_w*2:per_w*3],
                       x[...,per_w*3:per_w*4],
                       x[...,per_w*4:per_w*5],
                       x[...,per_w*5:],],
                       dim=1)
        x=x.reshape(-1,*x.shape[2:])

        embedding = self.conv_in(x)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)
        # wrong practice below
        # embedding = embedding.reshape(*embedding.shape[:-1],-1,6).permute(0,4,1,2,3) # bs,320,28,300 -> bs,6,320,28,50
        # embedding = embedding.reshape(-1,*embedding.shape[2:]) # bs,6,... -> bs*6,...
        return embedding