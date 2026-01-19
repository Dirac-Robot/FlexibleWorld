"""
Visualization utilities
"""
import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
import imageio


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array for visualization."""
    if tensor.dim() == 4:  # (B, C, H, W)
        tensor = tensor.permute(0, 2, 3, 1)
    elif tensor.dim() == 3:  # (C, H, W)
        tensor = tensor.permute(1, 2, 0)

    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr*255).astype(np.uint8)
    return arr


def save_video(
    frames: torch.Tensor,
    path: Union[str, Path],
    fps: int = 10,
) -> None:
    """
    Save frame sequence as video.

    Args:
        frames: (T, C, H, W) or (B, T, C, H, W) tensor
        path: output path (.mp4 or .gif)
        fps: frames per second
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if frames.dim() == 5:  # (B, T, C, H, W) - take first batch
        frames = frames[0]

    # Convert to numpy
    frames_np = []
    for t in range(frames.shape[0]):
        frame = tensor_to_numpy(frames[t])
        frames_np.append(frame)

    # Save
    if path.suffix == '.gif':
        imageio.mimsave(path, frames_np, fps=fps, loop=0)
    else:
        imageio.mimsave(path, frames_np, fps=fps)


def make_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        images: (N, C, H, W) tensor
        nrow: number of images per row
        padding: padding between images
        pad_value: value for padding

    Returns:
        (C, H', W') grid tensor
    """
    n, c, h, w = images.shape
    ncol = (n + nrow - 1)//nrow

    grid_h = ncol*h + (ncol + 1)*padding
    grid_w = nrow*w + (nrow + 1)*padding

    grid = torch.full((c, grid_h, grid_w), pad_value, device=images.device)

    for idx in range(n):
        row = idx//nrow
        col = idx%nrow
        y = padding + row*(h + padding)
        x = padding + col*(w + padding)
        grid[:, y:y+h, x:x+w] = images[idx]

    return grid


def save_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    path: Union[str, Path],
) -> None:
    """
    Save side-by-side comparison of original and reconstructed images.

    Args:
        original: (B, C, H, W) or (T, C, H, W) original images
        reconstructed: same shape as original
        path: output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Interleave for comparison
    n = original.shape[0]
    combined = torch.stack([original, reconstructed], dim=1)
    combined = combined.reshape(-1, *original.shape[1:])

    grid = make_grid(combined, nrow=min(8, n*2))
    img = tensor_to_numpy(grid)

    imageio.imwrite(path, img)
