"""
Interactive demo with keyboard controls
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from loguru import logger

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from config import scope
from inference import WorldModelAgent


# Default key mappings for common games
KEY_MAPPINGS = {
    'arrows': {
        'w': 0, 'up': 0,      # up
        's': 1, 'down': 1,    # down
        'a': 2, 'left': 2,    # left
        'd': 3, 'right': 3,   # right
    },
    'atari': {
        'w': 2,  # up
        's': 5,  # down
        'a': 4,  # left
        'd': 3,  # right
        ' ': 1,  # fire
    },
}


def get_keyboard_action(key_mapping: Dict[str, int], default: int = 0) -> int:
    """
    Get action from keyboard input (blocking).
    Uses simple input() for terminal-based interaction.
    """
    try:
        key = input('Action (wasd/arrows, q to quit): ').strip().lower()
        if key == 'q':
            return -1  # quit signal
        return key_mapping.get(key, default)
    except (KeyboardInterrupt, EOFError):
        return -1


@scope.observe(priority=1)
def demo_mode(config):
    """Demo mode configuration."""
    config.demo.model_path = None
    config.demo.from_hub = False
    config.demo.output_path = 'demo_output.gif'
    config.demo.max_steps = 100
    config.demo.key_mapping = 'arrows'
    config.demo.fps = 10


@scope
def demo(config):
    """Interactive demo with keyboard controls."""
    if config.demo.model_path is None:
        raise ValueError('Must set demo.model_path via CLI')

    # Load model
    if config.demo.from_hub:
        logger.info(f'Loading from HuggingFace Hub: {config.demo.model_path}')
        agent = WorldModelAgent.from_hub(config.demo.model_path)
    else:
        logger.info(f'Loading from checkpoint: {config.demo.model_path}')
        agent = WorldModelAgent.from_checkpoint(config.demo.model_path)

    # Get key mapping
    if isinstance(config.demo.key_mapping, str):
        key_mapping = KEY_MAPPINGS.get(config.demo.key_mapping, KEY_MAPPINGS['arrows'])
    else:
        key_mapping = dict(config.demo.key_mapping)

    logger.info('Starting interactive demo...')
    logger.info('Controls: WASD or arrow keys, Q to quit')
    logger.info(f'Key mapping: {key_mapping}')

    # Reset agent
    agent.reset()

    # Interactive loop
    step = 0
    while step < config.demo.max_steps:
        action = get_keyboard_action(key_mapping)

        if action == -1:
            logger.info('Quit signal received')
            break

        frame = agent.step(action)
        step += 1
        logger.info(f'Step {step}: action={action}')

    # Save output
    output_path = Path(config.demo.output_path)
    agent.save_history(output_path, fps=config.demo.fps)
    logger.info(f'Demo complete! Saved to {output_path}')


def gym_demo(
    env_name: str,
    model_path: str,
    num_episodes: int = 5,
    max_steps: int = 500,
    from_hub: bool = False,
    render: bool = True,
) -> None:
    """
    Run demo in a Gymnasium environment for comparison.

    Args:
        env_name: Gymnasium environment name
        model_path: checkpoint path or HuggingFace repo ID
        num_episodes: number of episodes to run
        max_steps: max steps per episode
        from_hub: whether to load from HuggingFace Hub
        render: whether to render environment
    """
    if not GYM_AVAILABLE:
        raise RuntimeError('gymnasium not installed')

    env = gym.make(env_name, render_mode='human' if render else None)

    if from_hub:
        agent = WorldModelAgent.from_hub(model_path)
    else:
        agent = WorldModelAgent.from_checkpoint(model_path)

    for episode in range(num_episodes):
        obs, info = env.reset()

        # Convert observation to tensor
        obs_tensor = torch.from_numpy(obs).float()
        if obs_tensor.dim() == 3 and obs_tensor.shape[-1] in [1, 3]:
            obs_tensor = obs_tensor.permute(2, 0, 1)
        if obs_tensor.max() > 1:
            obs_tensor = obs_tensor/255.0

        agent.reset(obs_tensor)

        total_reward = 0
        for step in range(max_steps):
            # Random action (could be replaced with policy)
            action = env.action_space.sample()

            # Step both environment and world model
            obs, reward, terminated, truncated, info = env.step(action)
            imagined_frame = agent.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        logger.info(f'Episode {episode+1}: reward={total_reward:.2f}, steps={step+1}')

        # Save comparison
        agent.save_history(f'gym_demo_ep{episode+1}.gif')

    env.close()


if __name__ == '__main__':
    demo()
