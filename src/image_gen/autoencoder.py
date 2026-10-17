import torch
import torch.nn as nn
import torch.nn.functional as F

def get_num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(channels, max_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1

class ResidualBlock(nn.Module):

    def __init__(
            self,
            in_channels:int,
            out_channels:int
            ):

        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(get_num_groups(in_channels),in_channels),
            nn.SiLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(
                get_num_groups(out_channels),
                out_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels,out_channels,1)
        )

    def forward(self,x:torch.Tensor) ->torch.Tensor:
        return self.skip(x) + self.block(x) * 0.5

class ImageEncoder(nn.Module):

    def __init__(self,latent_channels:int = 4,base_channels:int = 32):

        super().__init__()

        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d( 3, c, kernel_size= 3, padding = 1),
            ResidualBlock( c , c),

            nn.Conv2d( c, c * 2, 4, stride = 2, padding = 1),
            ResidualBlock(c * 2, c * 2),

            nn.Conv2d( c * 2, c * 4, 4, stride = 2, padding = 1),
            ResidualBlock( c * 4, c * 4 ),

            nn.Conv2d(c * 4, c * 4, kernel_size=4, stride=2, padding=1),
            ResidualBlock(c * 4, c * 4),

            nn.GroupNorm(get_num_groups(c * 4), c * 4),
            nn.SiLU(),
            nn.Conv2d(c * 4, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ImageDecoder(nn.Module):

    def __init__(self,latent_channels:int = 4,base_channels:int = 32):

        super().__init__()

        c = base_channels
        self.input = nn.Conv2d(latent_channels,c*4,3,padding=1)
        self.blocks = nn.Sequential(
            ResidualBlock( c * 4, c * 4),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(c*4,c*4,3,padding=1),
            ResidualBlock(c*4,c*2),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d( c * 2, c * 2, 3, padding = 1),
            ResidualBlock(c * 2, c),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(c,c,3,padding=1),
            ResidualBlock(c,c),
            nn.GroupNorm(8,c),
            nn.SiLU(),
            nn.Conv2d(c,3,3,padding=1),
            nn.Tanh(),
        )

    def forward(self,z:torch.Tensor)->torch.Tensor:
        return self.blocks(self.input(z))

class ImageAutoEncoder(nn.Module):
    """Small train from batch image autoencoder
        Input/output: RGB normalized to [-1,1].
        For [512x512] inputs, latent shape is [B,64,64,64].
    """

    def __init__(
            self,
            latent_channels:int = 4,
            base_channels:int = 32
            ):
        
        super().__init__()
        self.encoder = ImageEncoder(latent_channels,base_channels)
        self.decoder = ImageDecoder(latent_channels,base_channels)

    def encode(self,images:torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self,latents:torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self,images:torch.Tensor) -> dict[str,torch.Tensor]:
        latents = self.encode(images)
        reconstruction = self.decode(latents)

        return {
            "latents":latents,
            "reconstruction": reconstruction,
            "loss":F.l1_loss(reconstruction,images)
        }

    