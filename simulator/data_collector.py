"""
Data Collector for simulation data generation
"""
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
import json

from simulator.core import ParticleSimulator
from simulator.worlds.base import BaseWorld
from simulator.actions import ActionSpace, ActionType


class DataCollector:
    """Collect simulation data for world model training"""

    def __init__(self, output_dir: str = 'data/simulator'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_episode(self, simulator: ParticleSimulator,
                        policy: Optional[Callable] = None,
                        n_steps: int = 100,
                        render_scale: int = 1) -> dict:
        """
        Collect one episode of simulation data.

        Args:
            simulator: ParticleSimulator instance (should be initialized with particles)
            policy: Optional action policy function(state) -> action. Uses random if None.
            n_steps: Number of simulation steps
            render_scale: Scale factor for rendered images

        Returns:
            Episode data dict with states, actions, next_states, world_config
        """
        states = []
        actions = []
        next_states = []

        for step in range(n_steps):
            # render current state
            state_img = simulator.render(scale=render_scale)

            # get action
            if policy is not None:
                action = policy(simulator.get_state())
            else:
                action = ActionSpace.sample()

            # step simulation
            simulator.step(action=action)

            # render next state
            next_state_img = simulator.render(scale=render_scale)

            # store
            states.append(state_img)
            actions.append(action)
            next_states.append(next_state_img)

        return {
            'states': np.stack(states),          # (T, H, W, 3)
            'actions': np.array(actions),        # (T,)
            'next_states': np.stack(next_states),# (T, H, W, 3)
            'world_config': simulator.world.config.to_dict(),
            'world_vector': simulator.world.config.to_vector(),
        }

    def collect_dataset(self, world_factory: Callable[[], BaseWorld],
                        n_episodes: int = 100,
                        n_steps_per_episode: int = 100,
                        n_particles_range: tuple = (5, 20),
                        image_size: int = 64,
                        render_scale: int = 1,
                        policy: Optional[Callable] = None) -> List[dict]:
        """
        Collect full dataset with multiple episodes.

        Args:
            world_factory: Function that creates a BaseWorld instance
            n_episodes: Number of episodes to collect
            n_steps_per_episode: Steps per episode
            n_particles_range: (min, max) particles per episode
            image_size: Simulator resolution
            render_scale: Render scale factor
            policy: Optional policy function

        Returns:
            List of episode data dicts
        """
        episodes = []

        for ep_idx in range(n_episodes):
            # create world and simulator
            world = world_factory()
            sim = ParticleSimulator(
                world=world,
                width=image_size,
                height=image_size,
            )

            # add random particles
            n_particles = np.random.randint(*n_particles_range)
            for _ in range(n_particles):
                x = np.random.uniform(5, image_size-5)
                y = np.random.uniform(5, image_size-5)
                vx = np.random.uniform(-1, 1)
                vy = np.random.uniform(-1, 1)
                sim.add_particle(x=x, y=y, vx=vx, vy=vy)

            # add agent
            sim.add_agent(x=image_size//2, y=image_size//2)

            # collect episode
            episode_data = self.collect_episode(
                simulator=sim,
                policy=policy,
                n_steps=n_steps_per_episode,
                render_scale=render_scale,
            )
            episode_data['episode_idx'] = ep_idx
            episodes.append(episode_data)

            if (ep_idx+1) % 10 == 0:
                print(f'Collected {ep_idx+1}/{n_episodes} episodes')

        return episodes

    def save_dataset(self, episodes: List[dict], name: str = 'train'):
        """Save dataset to disk"""
        save_path = self.output_dir/f'{name}.npz'

        # stack all episodes
        all_states = np.concatenate([ep['states'] for ep in episodes])
        all_actions = np.concatenate([ep['actions'] for ep in episodes])
        all_next_states = np.concatenate([ep['next_states'] for ep in episodes])
        all_world_vectors = np.stack([ep['world_vector'] for ep in episodes
                                      for _ in range(len(ep['actions']))])

        # save
        np.savez_compressed(
            save_path,
            states=all_states,
            actions=all_actions,
            next_states=all_next_states,
            world_vectors=all_world_vectors,
        )

        # save world configs as JSON
        configs = [ep['world_config'] for ep in episodes]
        with open(self.output_dir/f'{name}_configs.json', 'w') as f:
            json.dump(configs, f, indent=2)

        print(f'Saved dataset to {save_path}')
        print(f'  States: {all_states.shape}')
        print(f'  Actions: {all_actions.shape}')
        print(f'  World vectors: {all_world_vectors.shape}')


if __name__ == '__main__':
    from simulator.worlds.basic_physics import BasicPhysicsWorld, ZeroGravityWorld

    collector = DataCollector(output_dir='data/simulator')

    # collect from basic physics world
    def random_world():
        gravity_y = np.random.choice([0.0, 0.05, 0.1, 0.15, -0.1])
        restitution = np.random.uniform(0.5, 1.0)
        return BasicPhysicsWorld(gravity_y=gravity_y, restitution=restitution)

    episodes = collector.collect_dataset(
        world_factory=random_world,
        n_episodes=10,
        n_steps_per_episode=50,
    )

    collector.save_dataset(episodes, name='test')
