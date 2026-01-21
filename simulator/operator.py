"""
Simulation Operator - reusable simulation scenarios and GIF generation
"""
from typing import Callable, List, Optional, Tuple
import numpy as np
from PIL import Image
import os

from simulator.core import ParticleSimulator
from simulator.worlds.base import BaseWorld
from simulator.worlds.basic_physics import BasicPhysicsWorld


class SimulationOperator:
    """Operator for running simulation scenarios and generating GIFs"""

    def __init__(self, sim: Optional[ParticleSimulator] = None,
                 width: int = 128, height: int = 128):
        self.sim = sim or ParticleSimulator(
            world=BasicPhysicsWorld(gravity_y=0),
            width=width, height=height
        )
        self.frames: List[Image.Image] = []
        self.output_dir = '/workspace/Projects/FlexibleWorld/outputs'

    def configure(self, em_force: float = 5.0, substeps: int = 4,
                  enable_heat: bool = True):
        """Configure simulation parameters"""
        self.sim.em_force_strength = em_force
        self.sim.substeps = substeps
        if not enable_heat:
            self.sim.vibration_scale = 0.0
            self.sim.thermal_radiation_rate = 0.0
            self.sim.heat_dissipation = 0.0
        return self

    def create_grid(self, rows: int = 4, cols: int = 4,
                    spacing: float = 6.0, offset: Tuple[float, float] = (24, 52),
                    charge_strength: float = 1.5, checkerboard: bool = True):
        """Create a grid of particles"""
        for row in range(rows):
            for col in range(cols):
                x = offset[0] + col*spacing
                y = offset[1] + row*spacing
                if checkerboard:
                    charge = charge_strength if (row + col) % 2 == 0 else -charge_strength
                else:
                    charge = charge_strength
                color = (0.9, 0.3, 0.3) if charge > 0 else (0.3, 0.3, 0.9)
                self.sim.add_particle(x=x, y=y, charge=charge, color=color, radius=2.0)
        return self

    def run_phase(self, steps: int, action: Optional[Callable[[int], None]] = None,
                  capture: bool = True, scale: int = 4):
        """Run simulation phase with optional per-step action"""
        for step in range(steps):
            if action:
                action(step)
            if capture:
                self.frames.append(Image.fromarray(self.sim.render(scale=scale)))
            self.sim.step()
        return self

    def apply_force(self, direction: Tuple[float, float], strength: float,
                    duration: int, ramp_down: bool = False):
        """Apply force to all particles"""
        def force_action(step):
            if ramp_down and step >= duration//2:
                factor = 1 - (step - duration//2)/(duration//2)
            else:
                factor = 1.0 if step < duration else 0.0
            for i in range(self.sim.n_particles):
                if self.sim.active[i]:
                    self.sim.velocities[i][0] += direction[0]*strength*factor
                    self.sim.velocities[i][1] += direction[1]*strength*factor
        return force_action

    def wall_repulsion(self, wall_x: float = 110, strength: float = 2.0,
                       range_dist: float = 15):
        """Create wall repulsion action"""
        def wall_action(step):
            for i in range(self.sim.n_particles):
                if self.sim.active[i]:
                    dist = wall_x - self.sim.positions[i][0]
                    if 0 < dist < range_dist:
                        self.sim.velocities[i][0] -= strength/(dist + 1)
        return wall_action

    def heat_center(self, x: float = 64, y: float = 64,
                    radius: float = 25, amount: float = 0.3):
        """Create heating action at center"""
        def heat_action(step):
            self.sim.add_heat_at(x, y, radius, amount)
        return heat_action

    def save_gif(self, name: str, duration: int = 35):
        """Save captured frames as GIF"""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f'{name}.gif')
        if self.frames:
            self.frames[0].save(
                path, save_all=True, append_images=self.frames[1:],
                duration=duration, loop=0
            )
            print(f'Saved: {path} ({len(self.frames)} frames)')
        return path

    def reset(self):
        """Reset simulation and frames"""
        self.sim.reset()
        self.frames.clear()
        return self


# Preset scenarios
def flubber_scenario(em_force: float = 8.0, charge: float = 2.0,
                     push_strength: float = 0.4) -> str:
    """Run flubber (elastic blob) scenario"""
    op = SimulationOperator()
    op.configure(em_force=em_force, enable_heat=False)
    op.create_grid(4, 4, spacing=5, charge_strength=charge)

    # Stabilize
    op.run_phase(50)

    # Push + wall collision
    wall = op.wall_repulsion(wall_x=105, strength=2.5)
    push = op.apply_force((1, 0), push_strength, duration=35)

    def combined(step):
        push(step)
        wall(step)
    op.run_phase(100, action=combined)

    # Recovery
    op.run_phase(100, action=wall)

    return op.save_gif('flubber_preset')


def heating_scenario(em_force: float = 2.0) -> str:
    """Run heating + dissociation scenario"""
    op = SimulationOperator()
    op.configure(em_force=em_force, enable_heat=True)
    op.create_grid(5, 5, spacing=8, offset=(44, 44))

    # Stabilize
    op.run_phase(50)

    # Heat center
    heat = op.heat_center(64, 64, radius=30, amount=0.4)
    op.run_phase(100, action=heat)

    # Observe dissociation
    op.run_phase(50)

    return op.save_gif('heating_preset')
