"""
WorldModel: Unified interface for encoder, dynamics, and decoder
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List

from .encoder import Encoder
from .decoder import Decoder
from .dynamics import RSSMDynamics


class WorldModel(nn.Module):
    """
    Complete World Model combining encoder, RSSM dynamics, and decoder.

    Provides high-level API for:
    - Training: encoding observations, dynamics prediction, reconstruction
    - Inference: imagination rollouts, frame rendering
    - Module replacement: swap encoder/decoder/dynamics
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        action_dim: int = 4,
        image_size: int = 64,
        channels: int = 3,
        encoder_channels: List[int] = None,
        decoder_channels: List[int] = None,
        discrete_actions: bool = True,
    ):
        super().__init__()

        encoder_channels = encoder_channels or [64, 128, 256, 512]
        decoder_channels = decoder_channels or [512, 256, 128, 64]

        self.encoder = Encoder(
            image_size=image_size,
            channels=channels,
            hidden_channels=encoder_channels,
            latent_dim=latent_dim,
        )

        self.dynamics = RSSMDynamics(
            hidden_dim=hidden_dim,
            stoch_dim=stoch_dim,
            stoch_classes=stoch_classes,
            action_dim=action_dim,
            embed_dim=latent_dim,
            discrete_actions=discrete_actions,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            stoch_dim=stoch_dim,
            stoch_classes=stoch_classes,
            image_size=image_size,
            channels=channels,
            hidden_channels=decoder_channels,
        )

        # Store config
        self._config = {
            'latent_dim': latent_dim,
            'hidden_dim': hidden_dim,
            'stoch_dim': stoch_dim,
            'stoch_classes': stoch_classes,
            'action_dim': action_dim,
            'image_size': image_size,
            'channels': channels,
        }

    def initial_state(self, batch_size: int, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Get initial RSSM state."""
        device = device or next(self.parameters()).device
        return self.dynamics.initial_state(batch_size, device)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to embedding."""
        return self.encoder(obs)

    def decode(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode state to image."""
        return self.decoder(state)

    def step(
        self,
        state: Dict[str, torch.Tensor],
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Single training step: observe → dynamics → decode.

        Args:
            state: current RSSM state
            obs: (B, C, H, W) observation
            action: (B,) action

        Returns:
            posterior: updated state using observation
            prior: predicted state (for KL loss)
            recon: reconstructed observation
        """
        embed = self.encode(obs)
        posterior, prior = self.dynamics.observe_step(state, action, embed)
        recon = self.decode(posterior)
        return posterior, prior, recon

    def imagine_step(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Imagination step: predict next state and render frame.

        Args:
            state: current RSSM state
            action: (B,) action

        Returns:
            next_state: predicted next state
            frame: rendered frame
        """
        next_state = self.dynamics.imagine_step(state, action)
        frame = self.decode(next_state)
        return next_state, frame

    def imagine(
        self,
        initial_state: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-step imagination rollout.

        Args:
            initial_state: starting RSSM state
            actions: (B, T) action sequence

        Returns:
            frames: (B, T, C, H, W) rendered frame sequence
        """
        batch_size, seq_len = actions.shape
        device = actions.device

        frames = []
        state = initial_state

        for t in range(seq_len):
            state, frame = self.imagine_step(state, actions[:, t])
            frames.append(frame)

        frames = torch.stack(frames, dim=1)
        return frames

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training on a sequence.

        Args:
            observations: (B, T, C, H, W) observation sequence
            actions: (B, T) action sequence
            state: optional initial state

        Returns:
            dict with:
                - recons: (B, T, C, H, W) reconstructions
                - posterior_logits: for KL loss
                - prior_logits: for KL loss
        """
        batch_size, seq_len = observations.shape[:2]
        device = observations.device

        if state is None:
            state = self.initial_state(batch_size, device)

        recons = []
        posterior_logits = []
        prior_logits = []

        for t in range(seq_len):
            obs_t = observations[:, t]
            action_t = actions[:, t]

            posterior, prior, recon = self.step(state, obs_t, action_t)

            recons.append(recon)
            posterior_logits.append(posterior['logits'])
            prior_logits.append(prior['logits'])

            state = posterior

        return {
            'recons': torch.stack(recons, dim=1),
            'posterior_logits': torch.stack(posterior_logits, dim=1),
            'prior_logits': torch.stack(prior_logits, dim=1),
            'final_state': state,
        }

    # === Module Replacement API ===

    def get_module(self, name: str) -> nn.Module:
        """Get a submodule by name."""
        if name == 'encoder':
            return self.encoder
        elif name == 'decoder':
            return self.decoder
        elif name == 'dynamics':
            return self.dynamics
        else:
            raise ValueError(f'Unknown module: {name}. Choose from: encoder, decoder, dynamics')

    def set_module(self, name: str, module: nn.Module) -> None:
        """Replace a submodule."""
        if name == 'encoder':
            self.encoder = module
        elif name == 'decoder':
            self.decoder = module
        elif name == 'dynamics':
            self.dynamics = module
        else:
            raise ValueError(f'Unknown module: {name}. Choose from: encoder, decoder, dynamics')

    def freeze_module(self, name: str) -> None:
        """Freeze a module's parameters."""
        module = self.get_module(name)
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, name: str) -> None:
        """Unfreeze a module's parameters."""
        module = self.get_module(name)
        for param in module.parameters():
            param.requires_grad = True

    def get_config(self) -> dict:
        return self._config.copy()


def create_world_model(config) -> WorldModel:
    """Factory function to create WorldModel from config."""
    return WorldModel(
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        stoch_dim=config.model.stoch_dim,
        stoch_classes=config.model.stoch_classes,
        action_dim=config.model.action_dim,
        image_size=config.model.image_size,
        channels=config.model.channels,
        encoder_channels=list(config.model.encoder.channels),
        decoder_channels=list(config.model.decoder.channels),
    )
