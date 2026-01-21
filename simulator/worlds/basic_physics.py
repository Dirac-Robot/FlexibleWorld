"""
Basic Physics World - gravity + elastic collision
"""
import numpy as np
from simulator.worlds.base import BaseWorld, WorldConfig


class BasicPhysicsWorld(BaseWorld):
    """Standard physics world with configurable gravity and elastic collisions"""

    def __init__(self, gravity_x: float = 0.0, gravity_y: float = 0.1,
                 restitution: float = 0.8, friction: float = 0.1):
        super().__init__()
        self._config = WorldConfig(
            name='basic_physics',
            gravity_x=gravity_x,
            gravity_y=gravity_y,
            restitution=restitution,
            friction=friction,
        )

    def apply_forces(self, positions: np.ndarray, velocities: np.ndarray,
                     masses: np.ndarray) -> np.ndarray:
        """Apply gravity to all particles"""
        n = len(positions)
        accelerations = np.zeros((n, 2), dtype=np.float32)
        accelerations[:, 0] = self._config.gravity_x
        accelerations[:, 1] = self._config.gravity_y
        return accelerations

    def collision_response(self, p1_vel: np.ndarray, p2_vel: np.ndarray,
                           p1_mass: float, p2_mass: float,
                           normal: np.ndarray) -> tuple:
        """Elastic collision with restitution coefficient"""
        e = self._config.restitution
        total_mass = p1_mass + p2_mass

        # relative velocity along collision normal
        rel_vel = p1_vel - p2_vel
        rel_vel_normal = np.dot(rel_vel, normal)

        # skip if separating
        if rel_vel_normal > 0:
            return p1_vel, p2_vel

        # impulse magnitude
        j = -(1 + e)*rel_vel_normal/total_mass

        # apply impulse
        new_p1_vel = p1_vel + (j*p2_mass)*normal
        new_p2_vel = p2_vel - (j*p1_mass)*normal

        return new_p1_vel.astype(np.float32), new_p2_vel.astype(np.float32)


class ZeroGravityWorld(BasicPhysicsWorld):
    """Zero gravity environment"""

    def __init__(self, restitution: float = 1.0, friction: float = 0.0):
        super().__init__(gravity_x=0.0, gravity_y=0.0,
                         restitution=restitution, friction=friction)
        self._config.name = 'zero_gravity'


class InverseGravityWorld(BasicPhysicsWorld):
    """Gravity points upward"""

    def __init__(self, gravity_strength: float = 0.1,
                 restitution: float = 0.8, friction: float = 0.1):
        super().__init__(gravity_x=0.0, gravity_y=-gravity_strength,
                         restitution=restitution, friction=friction)
        self._config.name = 'inverse_gravity'
