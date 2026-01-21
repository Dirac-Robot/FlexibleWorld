"""
Base World class - abstract interface for different physics rules
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np


@dataclass
class WorldConfig:
    """Explicit world parameters (used for World Vector encoding)"""
    name: str = 'base'
    gravity_x: float = 0.0
    gravity_y: float = 0.1
    restitution: float = 0.8  # bounciness coefficient
    friction: float = 0.1
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_vector(self) -> np.ndarray:
        """Convert explicit params to numpy vector"""
        base = np.array([
            self.gravity_x,
            self.gravity_y,
            self.restitution,
            self.friction,
        ], dtype=np.float32)
        return base

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'gravity_x': self.gravity_x,
            'gravity_y': self.gravity_y,
            'restitution': self.restitution,
            'friction': self.friction,
            **self.custom_params,
        }


class BaseWorld(ABC):
    """Abstract base class for world physics rules"""

    def __init__(self):
        self._config = WorldConfig()

    @property
    def config(self) -> WorldConfig:
        return self._config

    @abstractmethod
    def apply_forces(self, positions: np.ndarray, velocities: np.ndarray,
                     masses: np.ndarray) -> np.ndarray:
        """
        Apply world forces (gravity, etc.) to particles.
        Returns: acceleration array of shape (n_particles, 2)
        """
        pass

    @abstractmethod
    def collision_response(self, p1_vel: np.ndarray, p2_vel: np.ndarray,
                           p1_mass: float, p2_mass: float,
                           normal: np.ndarray) -> tuple:
        """
        Compute post-collision velocities for two particles.
        Returns: (new_p1_vel, new_p2_vel)
        """
        pass

    def on_boundary_collision(self, position: np.ndarray, velocity: np.ndarray,
                              bounds: tuple) -> tuple:
        """
        Handle collision with world boundaries.
        Returns: (new_position, new_velocity)
        """
        min_x, min_y, max_x, max_y = bounds
        new_pos = position.copy()
        new_vel = velocity.copy()

        if new_pos[0] < min_x:
            new_pos[0] = min_x
            new_vel[0] *= -self._config.restitution
        elif new_pos[0] > max_x:
            new_pos[0] = max_x
            new_vel[0] *= -self._config.restitution

        if new_pos[1] < min_y:
            new_pos[1] = min_y
            new_vel[1] *= -self._config.restitution
        elif new_pos[1] > max_y:
            new_pos[1] = max_y
            new_vel[1] *= -self._config.restitution

        return new_pos, new_vel
