"""
Inference pipeline for interactive world model rollouts
"""
import torch
from pathlib import Path
from typing import Optional, Union, List
from loguru import logger

from models import WorldModel
from models.world_model import create_world_model
from utils import save_video
from utils.hub import create_from_pretrained


class WorldModelAgent:
    """
    Interactive agent for world model inference.

    Supports game-like action → frame rendering loop.
    """

    def __init__(
        self,
        model: WorldModel,
        device: torch.device = None,
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        self.state = None
        self.history = []

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        device: torch.device = None,
    ) -> 'WorldModelAgent':
        """Load agent from local checkpoint."""
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get('model_config', {})

        model = WorldModel(**config)
        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(model, device)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        device: torch.device = None,
        **kwargs,
    ) -> 'WorldModelAgent':
        """Load agent from HuggingFace Hub."""
        model = create_from_pretrained(repo_id, device=device, **kwargs)
        return cls(model, device)

    def reset(self, observation: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reset agent state.

        Args:
            observation: optional initial observation (C, H, W) or (1, C, H, W)

        Returns:
            Initial frame
        """
        self.state = self.model.initial_state(1, self.device)
        self.history = []

        if observation is not None:
            if observation.dim() == 3:
                observation = observation.unsqueeze(0)
            observation = observation.to(self.device)

            # Encode observation into state
            with torch.no_grad():
                embed = self.model.encode(observation)
                # Use zero action for initial step
                zero_action = torch.zeros(1, dtype=torch.long, device=self.device)
                self.state, _ = self.model.dynamics.observe_step(self.state, zero_action, embed)

            frame = observation.squeeze(0)
        else:
            # Decode initial state to frame
            with torch.no_grad():
                frame = self.model.decode(self.state).squeeze(0)

        self.history.append(frame.cpu())
        return frame

    def step(self, action: int) -> torch.Tensor:
        """
        Take action and return next frame.

        Args:
            action: discrete action index

        Returns:
            Next frame (C, H, W)
        """
        if self.state is None:
            raise RuntimeError('Must call reset() before step()')

        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

        with torch.no_grad():
            self.state, frame = self.model.imagine_step(self.state, action_tensor)
            frame = frame.squeeze(0)

        self.history.append(frame.cpu())
        return frame

    def rollout(self, actions: List[int]) -> torch.Tensor:
        """
        Execute sequence of actions.

        Args:
            actions: list of action indices

        Returns:
            Frames tensor (T, C, H, W)
        """
        frames = []
        for action in actions:
            frame = self.step(action)
            frames.append(frame)
        return torch.stack(frames)

    def save_history(self, path: Union[str, Path], fps: int = 10) -> None:
        """Save accumulated history as video."""
        if not self.history:
            logger.warning('No history to save')
            return

        frames = torch.stack(self.history)
        save_video(frames, path, fps=fps)
        logger.info(f'Saved {len(self.history)} frames to {path}')

    def get_history(self) -> torch.Tensor:
        """Get accumulated frame history."""
        return torch.stack(self.history) if self.history else None


def interactive_rollout(
    model_path: Union[str, Path],
    output_path: Union[str, Path] = 'rollout.gif',
    num_steps: int = 50,
    from_hub: bool = False,
) -> None:
    """
    Run interactive rollout with random actions (for testing).

    Args:
        model_path: checkpoint path or HuggingFace repo ID
        output_path: output video path
        num_steps: number of steps to run
        from_hub: whether model_path is a HuggingFace repo ID
    """
    import random

    if from_hub:
        agent = WorldModelAgent.from_hub(model_path)
    else:
        agent = WorldModelAgent.from_checkpoint(model_path)

    action_dim = agent.model._config['action_dim']

    agent.reset()
    for _ in range(num_steps):
        action = random.randint(0, action_dim - 1)
        agent.step(action)

    agent.save_history(output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Interactive world model rollout')
    parser.add_argument('model_path', help='Checkpoint path or HuggingFace repo ID')
    parser.add_argument('--output', '-o', default='rollout.gif', help='Output video path')
    parser.add_argument('--steps', '-n', type=int, default=50, help='Number of steps')
    parser.add_argument('--hub', action='store_true', help='Load from HuggingFace Hub')
    args = parser.parse_args()

    interactive_rollout(
        args.model_path,
        args.output,
        args.steps,
        args.hub,
    )
