import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_num_groups(channels:int, max_groups:int = 8) -> int:
    for groups in range(min(channels,max_groups),0,-1):
        if channels % groups == 0:
            return groups

    return 1

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self,dim:int):
        super().__init__()
        self.dim = dim

    def forward(self,timesteps:torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2

        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            ) 
            / max(half_dim - 1, 1)
        )

        angles = timesteps.float()[:,None] * frequencies[None,:]

        embedding = torch.cat([angles.sin(),angles.cos(),], dim = -1,)

        if self.dim % 2:
            embedding = F.pad(embedding,(0,1))

        return embedding

class TimeResidualBlock(nn.Module):

    def __init__(
            self,
            in_channels:int,
            out_channels:int,
            time_dim:int,
        ):

        super().__init__()

        self.norm1 = nn.GroupNorm(
            get_num_groups(in_channels),
            in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_proj = nn.Linear(time_dim,out_channels)

        self.norm2 = nn.GroupNorm(
            get_num_groups(out_channels),
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels,out_channels,1)
        )

    def forward(
            self,
            x:torch.Tensor,
            time_embedding: torch.Tensor,
            )->torch.Tensor:

        h = self.conv1(F.silu(self.norm1(x)))

        time_bias = self.time_proj(F.silu(time_embedding))
        h = h + time_bias[:,:,None,None]

        h = self.conv2(F.silu(self.norm2(h)))

        return (self.skip(x) + h) / math.sqrt(2.0)

class Downsample(nn.Module):

    def __init__(self,channels:int):

        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):

    def __init__(self,channels:int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x,scale_factor=2,mode="nearest")
        return self.conv(x)

class LatentDiffusionUNet(nn.Module):
    """Predict DDPM noise in a compressed image latent space"""

    def __init__(
            self,
            latent_channels:int = 4,
            base_channels:int = 64,
        ):
        super().__init__()
        b = base_channels
        time_dim = b * 4

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(b),
            nn.Linear(b,time_dim),
            nn.SiLU(),
            nn.Linear(time_dim,time_dim),
        )

        self.input_conv = nn.Conv2d(
            latent_channels,
            b,
            3,
            padding=1,
        )

        # 64 x 64
        self.down_block1 = TimeResidualBlock(b,b,time_dim)
        self.downsample1 = Downsample(b)

        # 32 x 32
        self.down_block2 = TimeResidualBlock(b, 2 * b,time_dim)
        self.downsample2 = Downsample( 2 * b)

        # 16 x 16
        self.mid_block1 = TimeResidualBlock( 2 * b, 4 * b,time_dim)
        self.mid_block2 = TimeResidualBlock( 4 * b, 4 * b,time_dim)

        # Back to 32 x 32 and 64 x 64
        self.upsample2 = Upsample( 4 * b)
        self.up_block2 = TimeResidualBlock( 6 * b, 2 * b,time_dim)

        self.upsample1 = Upsample( 2 * b)
        self.up_block1 = TimeResidualBlock( 3 * b, b,time_dim)

        self.output_norm = nn.GroupNorm(
            get_num_groups(b),b
        )

        self.output_conv = nn.Conv2d(
            b,
            latent_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
            self,
            x:torch.Tensor,
            timesteps:torch.Tensor
        )->torch.Tensor:

        if x.ndim != 4:
            raise ValueError(
                f"x must have shape [B,C,H,W]"
            )

        if timesteps.ndim != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError(
                "timesteps must have shape [B]"
            )

        if x.shape[-2] % 4 or x.shape[-1] % 4:
            raise ValueError(
                "Latent height and width must be divisible by 4"
            )

        time = self.time_embedding(timesteps)

        h = self.input_conv(x)

        skip1 = self.down_block1(h,time)
        h = self.downsample1(skip1)        

        skip2 = self.down_block2(h,time)
        h = self.downsample2(skip2)

        h = self.mid_block1(h,time)
        h = self.mid_block2(h,time)

        h = self.upsample2(h)
        h = torch.cat([h,skip2],dim=1)
        h = self.up_block2(h,time)

        h = self.upsample1(h)
        h = torch.cat([h,skip1],dim=1)
        h = self.up_block1(h,time)

        return self.output_conv(F.silu(self.output_norm(h)))

