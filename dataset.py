"""
Dataset for frame sequence loading
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Union, List
import numpy as np
import h5py
from loguru import logger


class FrameSequenceDataset(Dataset):
    """
    Dataset for loading (frames, actions) sequences.

    Supports:
    - HDF5 files with 'frames' and 'actions' datasets
    - NPY files with separate _frames.npy and _actions.npy
    - Directory of npz files
    """

    def __init__(
        self,
        path: Union[str, Path],
        seq_len: int = 16,
        image_size: int = 64,
        transform: Optional[callable] = None,
    ):
        self.path = Path(path)
        self.seq_len = seq_len
        self.image_size = image_size
        self.transform = transform

        self._load_data()

    def _load_data(self):
        """Load data from file(s)."""
        if self.path.suffix == '.h5' or self.path.suffix == '.hdf5':
            self._load_hdf5()
        elif self.path.suffix == '.npy':
            self._load_npy()
        elif self.path.is_dir():
            self._load_directory()
        else:
            raise ValueError(f'Unsupported data format: {self.path}')

    def _load_hdf5(self):
        """Load from HDF5 file."""
        self.h5_file = h5py.File(self.path, 'r')
        self.frames = self.h5_file['frames']  # (N, T, C, H, W) or (N, T, H, W, C)
        self.actions = self.h5_file['actions']  # (N, T)

        # Check if NHWC format
        if self.frames.shape[-1] in [1, 3, 4]:
            self.nhwc = True
        else:
            self.nhwc = False

        self.num_episodes = len(self.frames)
        logger.info(f'Loaded HDF5 with {self.num_episodes} episodes')

    def _load_npy(self):
        """Load from NPY files."""
        frames_path = self.path
        actions_path = self.path.parent/(self.path.stem.replace('_frames', '') + '_actions.npy')

        self.frames = np.load(frames_path, mmap_mode='r')
        self.actions = np.load(actions_path, mmap_mode='r')
        self.nhwc = self.frames.shape[-1] in [1, 3, 4]
        self.num_episodes = len(self.frames)
        logger.info(f'Loaded NPY with {self.num_episodes} episodes')

    def _load_directory(self):
        """Load from directory of npz files."""
        self.npz_files = sorted(self.path.glob('*.npz'))
        self.num_episodes = len(self.npz_files)
        self.nhwc = True  # assume NHWC
        logger.info(f'Found {self.num_episodes} npz files in directory')

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence.

        Returns:
            frames: (T, C, H, W) float tensor [0, 1]
            actions: (T,) long tensor
        """
        if hasattr(self, 'npz_files'):
            data = np.load(self.npz_files[idx])
            frames = data['frames']
            actions = data['actions']
        else:
            frames = self.frames[idx]
            actions = self.actions[idx]

        # Ensure numpy array
        if isinstance(frames, h5py.Dataset):
            frames = frames[:]
        if isinstance(actions, h5py.Dataset):
            actions = actions[:]

        # Convert to tensor
        frames = torch.from_numpy(np.array(frames)).float()
        actions = torch.from_numpy(np.array(actions)).long()

        # Handle NHWC → NCHW
        if self.nhwc:
            frames = frames.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        if frames.max() > 1.0:
            frames = frames/255.0

        # Random crop for sequence
        if frames.shape[0] > self.seq_len:
            start = np.random.randint(0, frames.shape[0] - self.seq_len)
            frames = frames[start:start+self.seq_len]
            actions = actions[start:start+self.seq_len]

        # Apply transform
        if self.transform is not None:
            frames = self.transform(frames)

        return frames, actions


def create_dataloader(
    path: Union[str, Path],
    batch_size: int = 32,
    seq_len: int = 16,
    image_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """Create DataLoader from path."""
    dataset = FrameSequenceDataset(
        path=path,
        seq_len=seq_len,
        image_size=image_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs,
    )
