"""
Encoder: Image → Latent representation
"""
import torch
import torch.nn as nn
from typing import List


class Encoder(nn.Module):
    """CNN-based encoder for image observations."""

    def __init__(
        self,
        image_size: int = 64,
        channels: int = 3,
        hidden_channels: List[int] = [64, 128, 256, 512],
        latent_dim: int = 256,
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.latent_dim = latent_dim

        # Build CNN layers with proper spatial size tracking
        layers = []
        in_ch = channels
        spatial_size = image_size

        for i, out_ch in enumerate(hidden_channels):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            # Use GroupNorm (works for any batch size, simpler than LayerNorm for conv)
            layers.append(nn.GroupNorm(min(32, out_ch), out_ch))
            layers.append(nn.SiLU())
            in_ch = out_ch
            spatial_size //= 2

        self.conv = nn.Sequential(*layers)
        self.final_size = spatial_size
        self.flatten_dim = hidden_channels[-1]*self.final_size*self.final_size

        # Project to latent
        self.fc = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor

        Returns:
            embed: (B, latent_dim) embedding
        """
        batch_size = x.shape[0]
        h = self.conv(x)
        h = h.reshape(batch_size, -1)
        embed = self.fc(h)
        return embed

    def get_config(self) -> dict:
        return {
            'image_size': self.image_size,
            'channels': self.channels,
            'latent_dim': self.latent_dim,
        }
