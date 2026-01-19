"""
Flexible Backbone System for World Model

Supports:
1. Any pretrained encoder (CLIP, DINOv2, VideoMAE, etc.)
2. Any LLM backbone (LLaMA, Mistral, etc.)
3. Layer-wise output extraction for custom manipulation
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Union
from abc import ABC, abstractmethod
from loguru import logger


class LayerOutputCollector:
    """
    Collects intermediate layer outputs during forward pass.
    Attach hooks to specified layers and retrieve outputs.
    """

    def __init__(self):
        self.outputs: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register(self, module: nn.Module, name: str) -> None:
        """Register a hook on a module to collect its output."""
        def hook(m, inp, out):
            # Handle tuple outputs (like transformer layers)
            if isinstance(out, tuple):
                self.outputs[name] = out[0]
            else:
                self.outputs[name] = out

        handle = module.register_forward_hook(hook)
        self._hooks.append(handle)

    def clear(self) -> None:
        """Clear collected outputs."""
        self.outputs.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Get output by layer name."""
        return self.outputs.get(name)

    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all collected outputs."""
        return self.outputs.copy()


class BackboneWrapper(nn.Module, ABC):
    """
    Abstract wrapper for pretrained backbones.
    Provides unified interface for different model types.
    """

    def __init__(self, model: nn.Module, layer_names: List[str] = None):
        super().__init__()
        self.model = model
        self.collector = LayerOutputCollector()
        self.layer_names = layer_names or []

        # Register hooks for specified layers
        if self.layer_names:
            self._register_layer_hooks()

    def _register_layer_hooks(self) -> None:
        """Register hooks for named layers."""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                self.collector.register(module, name)
                logger.debug(f'Registered hook on layer: {name}')

    @abstractmethod
    def forward(self, x: Any) -> torch.Tensor:
        """Forward pass returning main output."""
        pass

    def get_layer_outputs(self) -> Dict[str, torch.Tensor]:
        """Get intermediate layer outputs from last forward pass."""
        return self.collector.get_all()

    def clear_layer_outputs(self) -> None:
        """Clear collected layer outputs."""
        self.collector.clear()

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def get_layer_names(self) -> List[str]:
        """List all available layer names."""
        return [name for name, _ in self.model.named_modules() if name]


class VisionEncoderWrapper(BackboneWrapper):
    """
    Wrapper for vision encoders (CLIP, DINOv2, VideoMAE, etc.)

    Example usage:
        # CLIP
        from transformers import CLIPVisionModel
        clip = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
        encoder = VisionEncoderWrapper(clip, layer_names=['encoder.layers.11'])

        # DINOv2
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        encoder = VisionEncoderWrapper(dinov2, layer_names=['blocks.11'])
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str] = None,
        output_key: str = None,  # for HF models that return dicts
        pool_output: bool = True,
    ):
        super().__init__(model, layer_names)
        self.output_key = output_key
        self.pool_output = pool_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or (B, T, C, H, W) for video

        Returns:
            features: (B, D) or (B, T, D) embeddings
        """
        self.collector.clear()

        # Handle video input
        is_video = x.dim() == 5
        if is_video:
            B, T, C, H, W = x.shape
            x = x.reshape(B*T, C, H, W)

        # Forward through model
        out = self.model(x)

        # Extract output (handle HF model outputs)
        if self.output_key and hasattr(out, self.output_key):
            out = getattr(out, self.output_key)
        elif hasattr(out, 'last_hidden_state'):
            out = out.last_hidden_state
        elif hasattr(out, 'pooler_output') and self.pool_output:
            out = out.pooler_output

        # Pool if needed (CLS token or mean)
        if out.dim() == 3 and self.pool_output:
            out = out[:, 0]  # CLS token

        # Reshape back for video
        if is_video:
            out = out.reshape(B, T, -1)

        return out


class LLMBackboneWrapper(BackboneWrapper):
    """
    Wrapper for LLM backbones (LLaMA, Mistral, etc.)

    Example usage:
        from transformers import AutoModelForCausalLM
        llm = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
        backbone = LLMBackboneWrapper(
            llm,
            layer_names=['model.layers.15', 'model.layers.31'],
            input_projection=nn.Linear(vision_dim, llm_dim),
        )
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str] = None,
        input_projection: nn.Module = None,
        output_projection: nn.Module = None,
    ):
        super().__init__(model, layer_names)
        self.input_projection = input_projection
        self.output_projection = output_projection

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_key_values: Any = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: (B, T, D) input embeddings
            attention_mask: (B, T) attention mask
            past_key_values: optional KV cache

        Returns:
            hidden_states: (B, T, D) output embeddings
        """
        self.collector.clear()

        # Project inputs if needed
        if self.input_projection is not None:
            inputs_embeds = self.input_projection(inputs_embeds)

        # Forward through LLM
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = out.last_hidden_state

        # Project outputs if needed
        if self.output_projection is not None:
            hidden_states = self.output_projection(hidden_states)

        return hidden_states


class LayerManipulator:
    """
    Placeholder for layer output manipulation during training.

    Define custom processing for intermediate layer outputs.
    This is where you implement:
    - Feature alignment losses
    - Distillation objectives
    - Auxiliary predictions
    - Custom regularization
    """

    def __init__(self):
        self.processors: Dict[str, Callable] = {}

    def register_processor(
        self,
        layer_name: str,
        processor: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Register a processor for a specific layer's output.

        Args:
            layer_name: name of the layer
            processor: function that takes layer output and returns processed output
        """
        self.processors[layer_name] = processor

    def process(
        self,
        layer_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Process all collected layer outputs.

        Args:
            layer_outputs: dict of layer_name -> tensor

        Returns:
            processed outputs
        """
        processed = {}
        for name, tensor in layer_outputs.items():
            if name in self.processors:
                processed[name] = self.processors[name](tensor)
            else:
                processed[name] = tensor
        return processed

    def compute_auxiliary_losses(
        self,
        layer_outputs: Dict[str, torch.Tensor],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute auxiliary losses from layer outputs.
        Override this method to implement custom losses.

        Returns:
            dict of loss_name -> loss_value
        """
        # Placeholder - implement your custom losses here
        return {}


class VideoDecoderWrapper(BackboneWrapper):
    """
    Wrapper for video/image decoders (VAE decoders, diffusion decoders, etc.)

    Example usage:
        # Stable Diffusion VAE decoder
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
        decoder = VideoDecoderWrapper(vae.decoder, layer_names=['mid_block'])

        # COSMOS tokenizer decoder
        decoder = VideoDecoderWrapper(cosmos_decoder, is_video=True)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str] = None,
        is_video: bool = False,
        input_projection: nn.Module = None,
    ):
        super().__init__(model, layer_names)
        self.is_video = is_video
        self.input_projection = input_projection

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) or (B, T, D) latent codes

        Returns:
            decoded: (B, C, H, W) or (B, T, C, H, W) decoded frames
        """
        self.collector.clear()

        # Project if needed
        if self.input_projection is not None:
            z = self.input_projection(z)

        # Handle video
        is_sequence = z.dim() == 3
        if is_sequence and not self.is_video:
            B, T, D = z.shape
            z = z.reshape(B*T, D)

        # Reshape for conv decoder if needed (B, D) → (B, D, 1, 1)
        if z.dim() == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)

        # Decode
        out = self.model(z)

        # Reshape back
        if is_sequence and not self.is_video:
            _, C, H, W = out.shape
            out = out.reshape(B, T, C, H, W)

        return out


def create_vae_decoder(
    model_name: str = 'stabilityai/sd-vae-ft-mse',
    layer_names: List[str] = None,
    freeze: bool = True,
) -> VideoDecoderWrapper:
    """Create VAE decoder wrapper from diffusers."""
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(model_name)
        wrapper = VideoDecoderWrapper(vae.decoder, layer_names=layer_names)
        if freeze:
            wrapper.freeze()
        logger.info(f'Loaded VAE decoder: {model_name}')
        return wrapper
    except ImportError:
        logger.warning('diffusers not installed, returning placeholder')
        return _create_placeholder_decoder()


def _create_placeholder_decoder() -> VideoDecoderWrapper:
    """Create placeholder decoder for testing."""
    class PlaceholderDecoder(nn.Module):
        def __init__(self, in_dim: int = 768, out_channels: int = 3, out_size: int = 64):
            super().__init__()
            self.out_channels = out_channels
            self.out_size = out_size
            self.net = nn.Sequential(
                nn.ConvTranspose2d(in_dim, 256, 4, 1, 0),
                nn.SiLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.SiLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.SiLU(),
                nn.ConvTranspose2d(32, out_channels, 4, 2, 1),
            )

        def forward(self, x):
            return self.net(x)

    return VideoDecoderWrapper(PlaceholderDecoder())


# ============================================================
# Placeholder implementations for common backbones
# ============================================================

def create_clip_encoder(
    model_name: str = 'openai/clip-vit-base-patch32',
    layer_names: List[str] = None,
    freeze: bool = True,
) -> VisionEncoderWrapper:
    """Create CLIP vision encoder wrapper."""
    try:
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained(model_name)
        wrapper = VisionEncoderWrapper(model, layer_names=layer_names)
        if freeze:
            wrapper.freeze()
        logger.info(f'Loaded CLIP encoder: {model_name}')
        return wrapper
    except ImportError:
        logger.warning('transformers not installed, returning placeholder')
        return _create_placeholder_encoder()


def create_dinov2_encoder(
    model_name: str = 'dinov2_vitb14',
    layer_names: List[str] = None,
    freeze: bool = True,
) -> VisionEncoderWrapper:
    """Create DINOv2 encoder wrapper."""
    try:
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        wrapper = VisionEncoderWrapper(model, layer_names=layer_names)
        if freeze:
            wrapper.freeze()
        logger.info(f'Loaded DINOv2 encoder: {model_name}')
        return wrapper
    except Exception as e:
        logger.warning(f'Failed to load DINOv2: {e}, returning placeholder')
        return _create_placeholder_encoder()


def create_llm_backbone(
    model_name: str = 'meta-llama/Llama-2-7b-hf',
    layer_names: List[str] = None,
    freeze: bool = True,
    load_in_8bit: bool = False,
) -> LLMBackboneWrapper:
    """Create LLM backbone wrapper."""
    try:
        from transformers import AutoModelForCausalLM
        kwargs = {}
        if load_in_8bit:
            kwargs['load_in_8bit'] = True
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        wrapper = LLMBackboneWrapper(model, layer_names=layer_names)
        if freeze:
            wrapper.freeze()
        logger.info(f'Loaded LLM backbone: {model_name}')
        return wrapper
    except ImportError:
        logger.warning('transformers not installed, returning placeholder')
        return _create_placeholder_llm()


def _create_placeholder_encoder() -> VisionEncoderWrapper:
    """Create placeholder encoder for testing."""
    class PlaceholderEncoder(nn.Module):
        def __init__(self, out_dim: int = 768):
            super().__init__()
            self.out_dim = out_dim
            self.proj = nn.Linear(3*64*64, out_dim)

        def forward(self, x):
            B = x.shape[0]
            x = x.reshape(B, -1)
            return self.proj(x)

    return VisionEncoderWrapper(PlaceholderEncoder())


def _create_placeholder_llm() -> LLMBackboneWrapper:
    """Create placeholder LLM for testing."""
    class PlaceholderLLM(nn.Module):
        def __init__(self, hidden_dim: int = 768):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True)
                for _ in range(4)
            ])

        def forward(self, inputs_embeds, **kwargs):
            x = inputs_embeds
            hidden_states = []
            for layer in self.layers:
                x = layer(x)
                hidden_states.append(x)

            class Output:
                def __init__(self, h, hs):
                    self.last_hidden_state = h
                    self.hidden_states = hs

            return Output(x, hidden_states)

    return LLMBackboneWrapper(PlaceholderLLM())
