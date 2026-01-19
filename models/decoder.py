"""
Decoder: Latent → Image reconstruction
"""
import torch
import torch.nn as nn
from typing import List


class Decoder(nn.Module):
    """CNN-based decoder for image reconstruction."""

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        image_size: int = 64,
        channels: int = 3,
        hidden_channels: List[int] = [512, 256, 128, 64],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels

        # Input: concatenation of deterministic + stochastic state
        input_dim = hidden_dim + stoch_dim*stoch_classes

        # Calculate initial spatial size
        num_layers = len(hidden_channels)
        self.init_size = image_size//(2**num_layers)
        self.init_channels = hidden_channels[0]

        # Project from latent to spatial
        self.fc = nn.Linear(input_dim, self.init_channels*self.init_size*self.init_size)

        # Build transposed CNN layers
        layers = []
        in_ch = hidden_channels[0]
        for i, out_ch in enumerate(hidden_channels[1:]):
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
            ])
            in_ch = out_ch

        # Final layer to image
        layers.append(nn.ConvTranspose2d(in_ch, channels, kernel_size=4, stride=2, padding=1))

        self.deconv = nn.Sequential(*layers)

    def forward(self, state: dict) -> torch.Tensor:
        """
        Args:
            state: dict with 'deter' (B, hidden_dim) and 'stoch' (B, stoch_dim*stoch_classes)

        Returns:
            recon: (B, C, H, W) reconstructed image
        """
        deter = state['deter']
        stoch = state['stoch'].reshape(deter.shape[0], -1)
        h = torch.cat([deter, stoch], dim=-1)

        h = self.fc(h)
        h = h.reshape(-1, self.init_channels, self.init_size, self.init_size)
        recon = self.deconv(h)
        return recon

    def get_config(self) -> dict:
        return {
            'latent_dim': self.latent_dim,
            'image_size': self.image_size,
            'channels': self.channels,
        }
