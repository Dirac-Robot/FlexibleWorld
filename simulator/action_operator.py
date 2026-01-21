"""
Action-Based Simulation Operator for Model Learning
Provides discrete action space that can be learned by RL/IL models.
"""
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import os

from simulator.core import ParticleSimulator
from simulator.worlds.basic_physics import BasicPhysicsWorld


class ActionType(IntEnum):
    """Discrete action types for simulation control"""
    NOOP = 0              # do nothing
    ADD_PARTICLE = 1      # add a new particle
    SET_PROPERTY = 2      # modify particle property
    APPLY_HEAT = 3        # add heat at location
    APPLY_FORCE = 4       # apply force to particle/region
    APPLY_ATTRACTION = 5  # create attraction point
    APPLY_REPULSION = 6   # create repulsion point
    STEP = 7              # advance simulation


class PropertyType(IntEnum):
    """Property types for SET_PROPERTY action"""
    POSITION_X = 0
    POSITION_Y = 1
    VELOCITY_X = 2
    VELOCITY_Y = 3
    CHARGE = 4
    MASS = 5
    RADIUS = 6
    TEMPERATURE = 7


@dataclass
class Action:
    """
    Unified action representation for simulation control.
    
    For model learning, this can be encoded as:
    - action_type: int (0-7)
    - target: int (particle index or -1 for global)
    - params: float[4] (x, y, value, radius)
    """
    action_type: ActionType
    target: int = -1  # particle index, -1 for position-based
    x: float = 0.0
    y: float = 0.0
    value: float = 0.0
    radius: float = 10.0
    property_type: PropertyType = PropertyType.POSITION_X

    def to_vector(self) -> np.ndarray:
        """Convert to learnable vector representation"""
        return np.array([
            float(self.action_type),
            float(self.target),
            self.x,
            self.y,
            self.value,
            self.radius,
            float(self.property_type)
        ], dtype=np.float32)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'Action':
        """Create action from vector"""
        return cls(
            action_type=ActionType(int(vec[0])),
            target=int(vec[1]),
            x=float(vec[2]),
            y=float(vec[3]),
            value=float(vec[4]),
            radius=float(vec[5]),
            property_type=PropertyType(int(vec[6]))
        )


class ActionOperator:
    """
    Action-based simulation operator for model learning.
    
    Usage:
        op = ActionOperator()
        
        # Execute actions
        op.execute(Action(ActionType.ADD_PARTICLE, x=32, y=32, value=1.0))  # charge=1.0
        op.execute(Action(ActionType.APPLY_HEAT, x=64, y=64, value=2.0, radius=15))
        op.execute(Action(ActionType.STEP))
        
        # Get state for model
        state = op.get_state()  # dict with positions, charges, etc.
        frame = op.render()     # visual observation
    """

    def __init__(self, width: int = 128, height: int = 128):
        self.sim = ParticleSimulator(
            world=BasicPhysicsWorld(gravity_y=0),
            width=width, height=height
        )
        self.sim.em_force_strength = 5.0
        self.sim.substeps = 3
        self.sim.vibration_scale = 0.5

        self.width = width
        self.height = height
        self.action_history: List[Action] = []
        self.frames: List[Image.Image] = []

        # Temporary forces (cleared each step)
        self._temp_attractions: List[Tuple[float, float, float, float]] = []  # x, y, strength, radius
        self._temp_repulsions: List[Tuple[float, float, float, float]] = []

    def execute(self, action: Action) -> bool:
        """
        Execute a single action. Returns True if successful.
        """
        self.action_history.append(action)

        if action.action_type == ActionType.NOOP:
            return True

        elif action.action_type == ActionType.ADD_PARTICLE:
            # value = charge, radius in action.radius
            idx = self.sim.add_particle(
                x=action.x, y=action.y,
                charge=action.value,
                radius=max(1.0, min(5.0, action.radius)),  # clamp radius
                mass=1.0
            )
            return idx >= 0

        elif action.action_type == ActionType.SET_PROPERTY:
            if action.target < 0 or action.target >= self.sim.n_particles:
                return False
            return self._set_property(action.target, action.property_type, action.value)

        elif action.action_type == ActionType.APPLY_HEAT:
            self.sim.add_heat_at(action.x, action.y, action.radius, action.value)
            return True

        elif action.action_type == ActionType.APPLY_FORCE:
            # Apply force to particles in radius
            return self._apply_force_at(action.x, action.y, action.value, action.radius)

        elif action.action_type == ActionType.APPLY_ATTRACTION:
            self._temp_attractions.append((action.x, action.y, action.value, action.radius))
            return True

        elif action.action_type == ActionType.APPLY_REPULSION:
            self._temp_repulsions.append((action.x, action.y, action.value, action.radius))
            return True

        elif action.action_type == ActionType.STEP:
            self._apply_temp_forces()
            self.sim.step()
            self._temp_attractions.clear()
            self._temp_repulsions.clear()
            return True

        return False

    def _set_property(self, idx: int, prop: PropertyType, value: float) -> bool:
        """Set a particle property"""
        if prop == PropertyType.POSITION_X:
            self.sim.positions[idx][0] = value
        elif prop == PropertyType.POSITION_Y:
            self.sim.positions[idx][1] = value
        elif prop == PropertyType.VELOCITY_X:
            self.sim.velocities[idx][0] = value
        elif prop == PropertyType.VELOCITY_Y:
            self.sim.velocities[idx][1] = value
        elif prop == PropertyType.CHARGE:
            self.sim.charges[idx] = value
        elif prop == PropertyType.MASS:
            self.sim.masses[idx] = max(0.1, value)
        elif prop == PropertyType.RADIUS:
            self.sim.radii[idx] = max(1.0, min(5.0, value))
        elif prop == PropertyType.TEMPERATURE:
            self.sim.temperatures[idx] = max(0, value)
        return True

    def _apply_force_at(self, x: float, y: float, strength: float, radius: float) -> bool:
        """Apply outward force from a point"""
        for i in range(self.sim.n_particles):
            if not self.sim.active[i]:
                continue
            dx = self.sim.positions[i][0] - x
            dy = self.sim.positions[i][1] - y
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radius and dist > 0.1:
                factor = strength*(1 - dist/radius)
                self.sim.velocities[i][0] += factor*dx/dist
                self.sim.velocities[i][1] += factor*dy/dist
        return True

    def _apply_temp_forces(self):
        """Apply temporary attraction/repulsion forces"""
        for ax, ay, strength, radius in self._temp_attractions:
            for i in range(self.sim.n_particles):
                if not self.sim.active[i]:
                    continue
                dx = ax - self.sim.positions[i][0]
                dy = ay - self.sim.positions[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radius and dist > 0.1:
                    factor = strength*(1 - dist/radius)
                    self.sim.velocities[i][0] += factor*dx/dist
                    self.sim.velocities[i][1] += factor*dy/dist

        for rx, ry, strength, radius in self._temp_repulsions:
            for i in range(self.sim.n_particles):
                if not self.sim.active[i]:
                    continue
                dx = self.sim.positions[i][0] - rx
                dy = self.sim.positions[i][1] - ry
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radius and dist > 0.1:
                    factor = strength*(1 - dist/radius)
                    self.sim.velocities[i][0] += factor*dx/dist
                    self.sim.velocities[i][1] += factor*dy/dist

    def get_state(self) -> Dict:
        """Get current state for model observation"""
        return {
            'positions': self.sim.positions[:self.sim.n_particles].copy(),
            'velocities': self.sim.velocities[:self.sim.n_particles].copy(),
            'charges': self.sim.charges[:self.sim.n_particles].copy(),
            'masses': self.sim.masses[:self.sim.n_particles].copy(),
            'temperatures': self.sim.temperatures[:self.sim.n_particles].copy(),
            'n_particles': self.sim.n_particles,
            'step': self.sim.step_count,
        }

    def get_observation(self, scale: int = 1) -> np.ndarray:
        """Get visual observation (rendered frame)"""
        return self.sim.render(scale=scale)

    def render(self, scale: int = 4) -> Image.Image:
        """Render current state as PIL Image"""
        return Image.fromarray(self.sim.render(scale=scale))

    def capture_frame(self, scale: int = 4):
        """Capture current frame to internal buffer"""
        self.frames.append(self.render(scale))

    def save_gif(self, path: str, duration: int = 50):
        """Save captured frames as GIF"""
        if self.frames:
            self.frames[0].save(
                path, save_all=True, append_images=self.frames[1:],
                duration=duration, loop=0
            )
        return path

    def reset(self):
        """Reset simulation and clear history"""
        self.sim.reset()
        self.action_history.clear()
        self.frames.clear()
        self._temp_attractions.clear()
        self._temp_repulsions.clear()

    @staticmethod
    def action_space_size() -> Tuple[int, int]:
        """Return (num_action_types, action_vector_dim)"""
        return (len(ActionType), 7)


# Convenience functions for creating actions
def add_particle(x: float, y: float, charge: float = 0.0, radius: float = 2.0) -> Action:
    return Action(ActionType.ADD_PARTICLE, x=x, y=y, value=charge, radius=radius)

def apply_heat(x: float, y: float, amount: float, radius: float = 15.0) -> Action:
    return Action(ActionType.APPLY_HEAT, x=x, y=y, value=amount, radius=radius)

def apply_attraction(x: float, y: float, strength: float, radius: float = 20.0) -> Action:
    return Action(ActionType.APPLY_ATTRACTION, x=x, y=y, value=strength, radius=radius)

def apply_repulsion(x: float, y: float, strength: float, radius: float = 20.0) -> Action:
    return Action(ActionType.APPLY_REPULSION, x=x, y=y, value=strength, radius=radius)

def step() -> Action:
    return Action(ActionType.STEP)

def set_charge(target: int, value: float) -> Action:
    return Action(ActionType.SET_PROPERTY, target=target, value=value, 
                  property_type=PropertyType.CHARGE)
