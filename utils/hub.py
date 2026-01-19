"""
HuggingFace Hub integration for pretrained world models
"""
import torch
from pathlib import Path
from typing import Dict, Optional, Union
from loguru import logger

try:
    from huggingface_hub import hf_hub_download, HfApi
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning('huggingface_hub not installed. Hub features disabled.')


# Registry of known pretrained models with their configs
PRETRAINED_REGISTRY = {
    # Format: 'model_name': {'repo_id': '...', 'filename': '...', 'config': {...}}
    # Add community/official models here
}


def register_pretrained(
    name: str,
    repo_id: str,
    filename: str = 'model.pt',
    config: Optional[dict] = None,
) -> None:
    """
    Register a pretrained model from HuggingFace Hub.

    Args:
        name: short name for the model
        repo_id: HuggingFace repo ID (e.g., 'username/world-model-atari')
        filename: checkpoint filename in the repo
        config: model config overrides
    """
    PRETRAINED_REGISTRY[name] = {
        'repo_id': repo_id,
        'filename': filename,
        'config': config or {},
    }


def list_pretrained() -> Dict[str, dict]:
    """List all registered pretrained models."""
    return PRETRAINED_REGISTRY.copy()


def download_pretrained(
    name_or_repo: str,
    filename: str = 'model.pt',
    cache_dir: Optional[str] = None,
    revision: str = 'main',
) -> Path:
    """
    Download pretrained model from HuggingFace Hub.

    Args:
        name_or_repo: registered name or HuggingFace repo ID
        filename: checkpoint filename
        cache_dir: optional cache directory
        revision: git revision (branch/tag/commit)

    Returns:
        Path to downloaded checkpoint
    """
    if not HF_AVAILABLE:
        raise RuntimeError('huggingface_hub not installed. Run: pip install huggingface_hub')

    # Check if it's a registered name
    if name_or_repo in PRETRAINED_REGISTRY:
        info = PRETRAINED_REGISTRY[name_or_repo]
        repo_id = info['repo_id']
        filename = info.get('filename', filename)
    else:
        repo_id = name_or_repo

    logger.info(f'Downloading {filename} from {repo_id}...')

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        revision=revision,
    )

    logger.info(f'Downloaded to {path}')
    return Path(path)


def load_from_hub(
    name_or_repo: str,
    model: Optional[torch.nn.Module] = None,
    filename: str = 'model.pt',
    cache_dir: Optional[str] = None,
    device: torch.device = None,
    strict: bool = False,
) -> tuple:
    """
    Load pretrained model from HuggingFace Hub.

    Args:
        name_or_repo: registered name or HuggingFace repo ID
        model: optional model to load weights into (if None, returns config for creation)
        filename: checkpoint filename
        cache_dir: optional cache directory
        device: device to load to
        strict: strict state dict loading

    Returns:
        (model_or_state_dict, config): loaded model/state and config
    """
    path = download_pretrained(name_or_repo, filename, cache_dir)
    device = device or torch.device('cpu')

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    config = checkpoint.get('model_config', checkpoint.get('config', {}))

    if model is not None:
        # Load into existing model
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=strict,
        )
        if missing:
            logger.warning(f'Missing keys: {missing}')
        if unexpected:
            logger.warning(f'Unexpected keys: {unexpected}')
        return model, config
    else:
        # Return state dict and config for manual creation
        return checkpoint['model_state_dict'], config


def push_to_hub(
    model: torch.nn.Module,
    repo_id: str,
    filename: str = 'model.pt',
    config: Optional[dict] = None,
    commit_message: str = 'Upload world model checkpoint',
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """
    Push model checkpoint to HuggingFace Hub.

    Args:
        model: trained model
        repo_id: target HuggingFace repo ID
        filename: checkpoint filename
        config: optional config to save
        commit_message: commit message
        private: whether repo should be private
        token: HuggingFace token

    Returns:
        URL to uploaded file
    """
    if not HF_AVAILABLE:
        raise RuntimeError('huggingface_hub not installed. Run: pip install huggingface_hub')

    import tempfile

    api = HfApi()

    # Create repo if needed
    api.create_repo(repo_id, private=private, exist_ok=True, token=token)

    # Save checkpoint to temp file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config() if hasattr(model, 'get_config') else None,
        }
        if config:
            checkpoint['config'] = config
        torch.save(checkpoint, f.name)

        # Upload
        url = api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo=filename,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token,
        )

    logger.info(f'Pushed to {url}')
    return url


def create_from_pretrained(
    name_or_repo: str,
    filename: str = 'model.pt',
    cache_dir: Optional[str] = None,
    device: torch.device = None,
    **override_config,
):
    """
    Create WorldModel from pretrained checkpoint, using saved config.

    Args:
        name_or_repo: registered name or HuggingFace repo ID
        filename: checkpoint filename
        cache_dir: optional cache directory
        device: device to load to
        **override_config: config overrides

    Returns:
        WorldModel instance with pretrained weights
    """
    from models.world_model import WorldModel

    # Get state dict and config
    state_dict, config = load_from_hub(
        name_or_repo,
        model=None,
        filename=filename,
        cache_dir=cache_dir,
        device=device,
    )

    # Apply overrides
    config.update(override_config)

    # Create model with saved config
    model = WorldModel(**config)

    # Load weights (non-strict to allow architecture variations)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f'Missing keys (may need training): {missing}')
    if unexpected:
        logger.info(f'Unexpected keys (ignored): {unexpected}')

    if device:
        model = model.to(device)

    return model
