"""
Checkpoint utilities for saving/loading models
"""
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: Union[str, Path],
    config: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: WorldModel instance
        optimizer: optional optimizer state
        epoch: current epoch
        path: save path
        config: optional config dict to save
        extra: optional extra data
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config() if hasattr(model, 'get_config') else None,
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if config is not None:
        checkpoint['config'] = config

    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    logger.info(f'Saved checkpoint to {path}')


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None,
) -> Dict:
    """
    Load checkpoint.

    Args:
        path: checkpoint path
        model: optional model to load state into
        optimizer: optional optimizer to load state into
        device: device to load to

    Returns:
        checkpoint dict
    """
    path = Path(path)
    device = device or torch.device('cpu')

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded model from {path}')

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info('Loaded optimizer state')

    return checkpoint


def load_pretrained(
    path: Union[str, Path],
    model: torch.nn.Module,
    freeze_modules: List[str] = None,
    strict: bool = True,
    device: torch.device = None,
) -> Dict:
    """
    Load pretrained model for finetuning.

    Args:
        path: pretrained checkpoint path
        model: model to load into
        freeze_modules: list of module names to freeze ('encoder', 'decoder', 'dynamics')
        strict: whether to strictly enforce state dict matching
        device: device to load to

    Returns:
        checkpoint dict
    """
    checkpoint = load_checkpoint(path, model, device=device)

    freeze_modules = freeze_modules or []
    for module_name in freeze_modules:
        model.freeze_module(module_name)
        logger.info(f'Froze module: {module_name}')

    return checkpoint
