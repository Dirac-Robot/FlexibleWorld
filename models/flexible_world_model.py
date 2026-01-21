"""
Flexible World Model with pluggable encoder/LLM and layer manipulation
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable

from .backbone import (
    VisionEncoderWrapper,
    LLMBackboneWrapper,
    LayerManipulator,
    LayerOutputCollector,
    create_clip_encoder,
    create_dinov2_encoder,
    create_llm_backbone,
    _create_placeholder_encoder,
    _create_placeholder_llm,
)


class FlexibleWorldModel(nn.Module):
    """
    Flexible World Model with:
    1. Pluggable vision encoder (CLIP, DINOv2, etc.)
    2. Pluggable LLM backbone (LLaMA, Mistral, etc.)
    3. Layer-wise output extraction and manipulation
    4. Custom decoder

    Architecture:
        Image → VisionEncoder → Projection → LLMBackbone → Decoder → Next Frame
                   ↓                            ↓
              [layer outputs]              [layer outputs]
                   ↓                            ↓
              LayerManipulator           LayerManipulator
    """

    def __init__(
        self,
        vision_encoder: VisionEncoderWrapper = None,
        llm_backbone: LLMBackboneWrapper = None,
        decoder: nn.Module = None,
        action_dim: int = 4,
        vision_dim: int = 768,
        llm_dim: int = 768,
        output_dim: int = 768,
        image_size: int = 64,
        channels: int = 3,
    ):
        super().__init__()

        # Store dimensions
        self.action_dim = action_dim
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.output_dim = output_dim
        self.image_size = image_size
        self.channels = channels

        # Vision encoder (placeholder if not provided)
        self.vision_encoder = vision_encoder or _create_placeholder_encoder()

        # LLM backbone (placeholder if not provided)
        self.llm_backbone = llm_backbone or _create_placeholder_llm()

        # Projections
        self.vision_to_llm = nn.Linear(vision_dim, llm_dim)
        self.action_embed = nn.Embedding(action_dim, llm_dim)

        # Decoder (placeholder if not provided)
        self.decoder = decoder or self._create_default_decoder()

        # Layer manipulators
        self.vision_manipulator = LayerManipulator()
        self.llm_manipulator = LayerManipulator()

        # For custom layer processing during training
        self._custom_layer_processors: Dict[str, Callable] = {}

    def _create_default_decoder(self) -> nn.Module:
        """Create default CNN decoder."""
        return nn.Sequential(
            nn.Linear(self.output_dim, 512*4*4),
            nn.SiLU(),
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, self.channels, 4, 2, 1),
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        return_layer_outputs: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with layer output collection.

        Args:
            observations: (B, T, C, H, W) image sequence
            actions: (B, T) action indices
            return_layer_outputs: whether to return intermediate layer outputs

        Returns:
            dict with:
                - predictions: (B, T, C, H, W) predicted frames
                - vision_layer_outputs: optional dict of vision encoder layer outputs
                - llm_layer_outputs: optional dict of LLM layer outputs
        """
        B, T, C, H, W = observations.shape

        # === Vision Encoding ===
        # Encode all frames
        vision_features = self.vision_encoder(observations)  # (B, T, vision_dim)

        # Collect vision layer outputs
        vision_layer_outputs = self.vision_encoder.get_layer_outputs()

        # === Process through manipulator (placeholder) ===
        if vision_layer_outputs:
            vision_layer_outputs = self.vision_manipulator.process(vision_layer_outputs)

        # === Prepare LLM inputs ===
        # Project vision features
        llm_inputs = self.vision_to_llm(vision_features)  # (B, T, llm_dim)

        # Add action embeddings (shift by 1 for autoregressive)
        action_embeds = self.action_embed(actions)  # (B, T, llm_dim)

        # Interleave: [obs_0, act_0, obs_1, act_1, ...]
        # Or simply add action information
        llm_inputs = llm_inputs + action_embeds

        # === LLM backbone ===
        llm_outputs = self.llm_backbone(llm_inputs)  # (B, T, llm_dim)

        # Collect LLM layer outputs
        llm_layer_outputs = self.llm_backbone.get_layer_outputs()

        # Process through manipulator (placeholder)
        if llm_layer_outputs:
            llm_layer_outputs = self.llm_manipulator.process(llm_layer_outputs)

        # === Decode to frames ===
        # Reshape for decoding
        llm_outputs_flat = llm_outputs.reshape(B*T, -1)
        predictions_flat = self.decoder(llm_outputs_flat)
        predictions = predictions_flat.reshape(B, T, C, H, W)

        # === Build output ===
        output = {
            'predictions': predictions,
        }

        if return_layer_outputs:
            output['vision_layer_outputs'] = vision_layer_outputs
            output['llm_layer_outputs'] = llm_layer_outputs

        return output

    def imagine(
        self,
        initial_observation: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Autoregressive imagination rollout.

        Args:
            initial_observation: (B, C, H, W) starting frame
            actions: (B, T) action sequence

        Returns:
            frames: (B, T, C, H, W) imagined frames
        """
        B, T = actions.shape
        device = actions.device

        frames = []
        current_obs = initial_observation.unsqueeze(1)  # (B, 1, C, H, W)

        for t in range(T):
            action_t = actions[:, t:t+1]  # (B, 1)

            # Predict next frame
            out = self.forward(current_obs, action_t)
            next_frame = out['predictions'][:, -1]  # (B, C, H, W)

            frames.append(next_frame)
            current_obs = next_frame.unsqueeze(1)

        return torch.stack(frames, dim=1)

    # ============================================================
    # Module management API
    # ============================================================

    def set_vision_encoder(self, encoder: VisionEncoderWrapper) -> None:
        """Replace vision encoder."""
        self.vision_encoder = encoder

    def set_llm_backbone(self, backbone: LLMBackboneWrapper) -> None:
        """Replace LLM backbone."""
        self.llm_backbone = backbone

    def set_decoder(self, decoder: nn.Module) -> None:
        """Replace decoder."""
        self.decoder = decoder

    def get_vision_layer_names(self) -> List[str]:
        """Get available vision encoder layer names."""
        return self.vision_encoder.get_layer_names()

    def get_llm_layer_names(self) -> List[str]:
        """Get available LLM layer names."""
        return self.llm_backbone.get_layer_names()

    def register_vision_processor(
        self,
        layer_name: str,
        processor: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Register processor for vision encoder layer output."""
        self.vision_manipulator.register_processor(layer_name, processor)

    def register_llm_processor(
        self,
        layer_name: str,
        processor: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Register processor for LLM layer output."""
        self.llm_manipulator.register_processor(layer_name, processor)

    def freeze_vision_encoder(self) -> None:
        """Freeze vision encoder."""
        self.vision_encoder.freeze()

    def freeze_llm_backbone(self) -> None:
        """Freeze LLM backbone."""
        self.llm_backbone.freeze()

    def unfreeze_vision_encoder(self) -> None:
        """Unfreeze vision encoder."""
        self.vision_encoder.unfreeze()

    def unfreeze_llm_backbone(self) -> None:
        """Unfreeze LLM backbone."""
        self.llm_backbone.unfreeze()


def create_flexible_world_model(config) -> FlexibleWorldModel:
    """Factory function to create FlexibleWorldModel from config."""

    vision_encoder = None
    llm_backbone = None

    # Create vision encoder based on config
    encoder_type = config.model.vision.get('type', 'placeholder')
    if encoder_type == 'clip':
        vision_encoder = create_clip_encoder(
            model_name=config.model.vision.get('name', 'openai/clip-vit-base-patch32'),
            layer_names=list(config.model.vision.get('layer_names', [])),
            freeze=config.model.vision.get('freeze', True),
        )
    elif encoder_type == 'dinov2':
        vision_encoder = create_dinov2_encoder(
            model_name=config.model.vision.get('name', 'dinov2_vitb14'),
            layer_names=list(config.model.vision.get('layer_names', [])),
            freeze=config.model.vision.get('freeze', True),
        )

    # Create LLM backbone based on config
    llm_type = config.model.llm.get('type', 'placeholder')
    if llm_type in ['llama', 'mistral', 'hf']:
        llm_backbone = create_llm_backbone(
            model_name=config.model.llm.get('name', 'meta-llama/Llama-2-7b-hf'),
            layer_names=list(config.model.llm.get('layer_names', [])),
            freeze=config.model.llm.get('freeze', True),
            load_in_8bit=config.model.llm.get('load_in_8bit', False),
        )

    return FlexibleWorldModel(
        vision_encoder=vision_encoder,
        llm_backbone=llm_backbone,
        action_dim=config.model.action_dim,
        vision_dim=config.model.vision.get('dim', 768),
        llm_dim=config.model.llm.get('dim', 768),
        output_dim=config.model.decoder.get('output_dim', 768),
        image_size=config.model.image_size,
        channels=config.model.channels,
    )
