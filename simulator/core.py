"""
Core Particle Simulator using Taichi for GPU acceleration
"""
import taichi as ti
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from simulator.worlds.base import BaseWorld
from simulator.worlds.basic_physics import BasicPhysicsWorld
from simulator.actions import ActionType, ActionSpace, ACTION_DELTAS
from simulator.particle_types import PARTICLE_REGISTRY, ParticleType, Reaction


@dataclass
class Particle:
    """Particle data structure"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 2.0
    mass: float = 1.0
    particle_type: int = 0  # 0=normal, 1=agent
    temperature: float = 0.0  # heat level (0=cold, higher=hotter)
    thermal_conductivity: float = 0.5  # 0=insulator, 1=perfect conductor
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)


class ParticleSimulator:
    """
    2D Particle Simulator with configurable world physics.
    Uses NumPy for computation (Taichi integration for future optimization).
    """

    def __init__(self, world: Optional[BaseWorld] = None,
                 width: int = 64, height: int = 64,
                 max_particles: int = 256):
        self.width = width
        self.height = height
        self.max_particles = max_particles
        self.world = world if world is not None else BasicPhysicsWorld()

        # particle arrays
        self.positions = np.zeros((max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((max_particles, 2), dtype=np.float32)
        self.radii = np.zeros(max_particles, dtype=np.float32)
        self.masses = np.zeros(max_particles, dtype=np.float32)
        self.types = np.zeros(max_particles, dtype=np.int32)  # legacy int type (0=normal, 1=agent)
        self.type_names = [''] * max_particles  # string type names from registry
        self.temperatures = np.zeros(max_particles, dtype=np.float32)
        self.thermal_conductivities = np.zeros(max_particles, dtype=np.float32)
        self.colors = np.zeros((max_particles, 3), dtype=np.float32)
        self.active = np.zeros(max_particles, dtype=bool)

        # orientation and electromagnetic properties
        self.angles = np.zeros(max_particles, dtype=np.float32)  # rotation angle (radians)
        self.charges = np.zeros(max_particles, dtype=np.float32)  # em charge magnitude
        self.polarity_angles = np.zeros(max_particles, dtype=np.float32)  # direction of polarity

        # valence chemistry properties (for emergent bonding)
        self.valences = np.zeros(max_particles, dtype=np.int32)  # bonding capacity
        self.electronegativities = np.zeros(max_particles, dtype=np.float32)
        self.bond_energies = np.zeros(max_particles, dtype=np.float32)
        self.bond_radii = np.zeros(max_particles, dtype=np.float32)

        # heat simulation parameters
        self.heat_dissipation = 0.02  # ambient heat loss per step
        self.vibration_scale = 0.5  # temperature -> position jitter
        self.thermal_radiation_rate = 0.01  # heat radiated to nearby particles
        self.thermal_radiation_radius = 10.0  # radius for thermal radiation

        # electromagnetic force parameters
        self.em_force_strength = 0.5  # strength of attraction/repulsion
        self.em_force_radius = 15.0  # max distance for em force

        # bonding parameters
        self.enable_bonding = True  # use emergent bonding instead of reaction table
        self.min_bonding_energy = 0.5  # minimum kinetic energy for bonding

        # reaction parameters
        self.enable_reactions = True
        self.pending_reactions: List[Tuple[int, int, Reaction]] = []  # collected during step

        self.n_particles = 0
        self.agent_idx = -1  # index of agent particle

        # action space config
        self.action_space = ActionSpace()

        # simulation state
        self.step_count = 0
        self.substeps = 1  # subdivide each step for better collision detection
        self.dt = 1.0  # base time step (divided by substeps internally)

    def add_particle(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0,
                     radius: float = 2.0, mass: float = 1.0, particle_type: int = 0,
                     temperature: float = 0.0,
                     thermal_conductivity: float = 0.5,
                     angle: float = 0.0,
                     charge: float = 0.0,
                     polarity_angle: float = 0.0,
                     valence: int = 0,
                     electronegativity: float = 0.5,
                     bond_energy: float = 0.0,
                     bond_radius: float = 4.0,
                     color: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> int:
        """Add a particle to the simulation."""
        if self.n_particles >= self.max_particles:
            return -1

        idx = self.n_particles
        self.positions[idx] = [x, y]
        self.velocities[idx] = [vx, vy]
        self.radii[idx] = radius
        self.masses[idx] = mass
        self.types[idx] = particle_type
        self.temperatures[idx] = temperature
        self.thermal_conductivities[idx] = thermal_conductivity
        self.angles[idx] = angle
        self.charges[idx] = charge
        self.polarity_angles[idx] = polarity_angle
        self.valences[idx] = valence
        self.electronegativities[idx] = electronegativity
        self.bond_energies[idx] = bond_energy
        self.bond_radii[idx] = bond_radius
        self.colors[idx] = color
        self.active[idx] = True
        self.n_particles += 1
        return idx

    def add_typed_particle(self, x: float, y: float, type_name: str,
                           vx: float = 0.0, vy: float = 0.0,
                           temperature: Optional[float] = None,
                           charge: Optional[float] = None) -> int:
        """
        Add a particle from the registry by type name.
        Properties are loaded from ParticleRegistry.
        """
        ptype = PARTICLE_REGISTRY.get_type(type_name)
        if ptype is None:
            raise ValueError(f"Unknown particle type: {type_name}")

        temp = temperature if temperature is not None else ptype.base_temperature
        chrg = charge if charge is not None else ptype.charge

        idx = self.add_particle(
            x=x, y=y, vx=vx, vy=vy,
            radius=ptype.radius,
            mass=ptype.mass,
            particle_type=0,
            temperature=temp,
            thermal_conductivity=ptype.thermal_conductivity,
            charge=chrg,
            valence=ptype.valence,
            electronegativity=ptype.electronegativity,
            bond_energy=ptype.bond_energy,
            bond_radius=ptype.bond_radius,
            color=ptype.color,
        )
        if idx >= 0:
            self.type_names[idx] = type_name
        return idx

    def add_agent(self, x: float, y: float, radius: float = 3.0) -> int:
        """Add agent particle (controllable)"""
        idx = self.add_typed_particle(x=x, y=y, type_name='agent')
        if idx >= 0:
            self.types[idx] = 1  # mark as agent type
            self.agent_idx = idx
        return idx

    def remove_particle(self, idx: int):
        """Mark particle as inactive"""
        if 0 <= idx < self.n_particles:
            self.active[idx] = False
            if idx == self.agent_idx:
                self.agent_idx = -1

    def _apply_action(self, action: int):
        """Apply agent action"""
        if self.agent_idx < 0 or not self.active[self.agent_idx]:
            return

        agent_pos = self.positions[self.agent_idx]
        agent_vel = self.velocities[self.agent_idx]

        # movement actions
        if action in (ActionType.MOVE_UP, ActionType.MOVE_DOWN,
                      ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT):
            dx, dy = ActionSpace.get_move_delta(action, self.action_space.move_speed)
            self.velocities[self.agent_idx] = [dx, dy]

        elif action == ActionType.NOOP:
            # apply friction to slow down
            self.velocities[self.agent_idx] *= 0.8

        elif action == ActionType.PUSH:
            # push nearby particles away
            for i in range(self.n_particles):
                if i == self.agent_idx or not self.active[i]:
                    continue
                diff = self.positions[i] - agent_pos
                dist = np.linalg.norm(diff)
                if 0 < dist < self.action_space.interaction_radius:
                    direction = diff/dist
                    self.velocities[i] += direction*self.action_space.push_force

        elif action == ActionType.PULL:
            # pull nearby particles toward agent
            for i in range(self.n_particles):
                if i == self.agent_idx or not self.active[i]:
                    continue
                diff = agent_pos - self.positions[i]
                dist = np.linalg.norm(diff)
                if 0 < dist < self.action_space.interaction_radius:
                    direction = diff/dist
                    self.velocities[i] += direction*self.action_space.pull_force

        elif action == ActionType.SPAWN:
            # spawn particle in front of agent (based on last movement)
            spawn_offset = np.array([5.0, 0.0])  # default right
            spawn_pos = agent_pos + spawn_offset
            if 0 <= spawn_pos[0] < self.width and 0 <= spawn_pos[1] < self.height:
                self.add_particle(
                    x=spawn_pos[0], y=spawn_pos[1],
                    color=(0.8, 0.4, 0.2)  # orange
                )

    def _detect_collisions(self) -> List[Tuple[int, int]]:
        """Detect particle-particle collisions"""
        collisions = []
        for i in range(self.n_particles):
            if not self.active[i]:
                continue
            for j in range(i+1, self.n_particles):
                if not self.active[j]:
                    continue
                diff = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(diff)
                min_dist = self.radii[i] + self.radii[j]
                if dist < min_dist and dist > 0:
                    collisions.append((i, j))
        return collisions

    def _resolve_collision(self, i: int, j: int):
        """
        Resolve collision between two particles.
        Energy distribution:
        - restitution controls how much kinetic energy is preserved
        - (1 - restitution) of collision energy converts to heat
        - thermal_conductivity controls heat transfer rate between particles
        """
        diff = self.positions[j] - self.positions[i]
        dist = np.linalg.norm(diff)
        if dist == 0:
            return

        normal = diff/dist
        min_dist = self.radii[i] + self.radii[j]

        # separate particles
        overlap = min_dist - dist
        total_mass = self.masses[i] + self.masses[j]
        self.positions[i] -= normal*overlap*(self.masses[j]/total_mass)
        self.positions[j] += normal*overlap*(self.masses[i]/total_mass)

        # derive restitution from charge interaction
        # same sign charges → EM repulsion → high restitution (bouncy)
        # opposite sign charges → EM attraction → low restitution (sticky)
        q_i, q_j = self.charges[i], self.charges[j]
        charge_product = q_i*q_j  # positive = same sign, negative = opposite
        base_restitution = 0.5
        # restitution ranges from 0.2 (opposite charges) to 0.95 (same charges)
        effective_restitution = np.clip(base_restitution + 0.3*charge_product, 0.1, 0.98)

        # calculate relative velocity along collision normal
        rel_vel = self.velocities[i] - self.velocities[j]
        rel_vel_normal = np.dot(rel_vel, normal)

        # skip if separating (rel_vel_normal < 0 means moving apart)
        if rel_vel_normal < 0:
            return

        # calculate collision impulse with per-particle restitution
        j_impulse = -(1 + effective_restitution)*rel_vel_normal/total_mass

        # apply impulse to velocities
        self.velocities[i] += (j_impulse*self.masses[j])*normal
        self.velocities[j] -= (j_impulse*self.masses[i])*normal

        # energy lost in collision -> convert to heat
        # energy_lost = 0.5 * reduced_mass * (1 - e^2) * rel_vel_normal^2
        reduced_mass = (self.masses[i]*self.masses[j])/total_mass
        energy_lost = 0.5*reduced_mass*(1 - effective_restitution**2)*(rel_vel_normal**2)

        # distribute heat based on mass ratio (lighter particle heats more)
        heat_i = energy_lost*(self.masses[j]/total_mass)*0.5  # scale factor
        heat_j = energy_lost*(self.masses[i]/total_mass)*0.5
        self.temperatures[i] += heat_i
        self.temperatures[j] += heat_j

        # heat conduction between particles (based on thermal conductivity)
        cond_i, cond_j = self.thermal_conductivities[i], self.thermal_conductivities[j]
        effective_conductivity = 2*cond_i*cond_j/(cond_i + cond_j + 1e-8)  # harmonic mean

        temp_i, temp_j = self.temperatures[i], self.temperatures[j]
        heat_flow = (temp_j - temp_i)*effective_conductivity*0.5
        self.temperatures[i] += heat_flow
        self.temperatures[j] -= heat_flow

        # check for chemical reaction
        if self.enable_reactions:
            type_i = self.type_names[i]
            type_j = self.type_names[j]
            if type_i and type_j:
                reaction = PARTICLE_REGISTRY.get_reaction(type_i, type_j)
                if reaction is not None:
                    # collision energy = kinetic energy before collision
                    collision_energy = 0.5*reduced_mass*(rel_vel_normal**2)
                    if collision_energy >= reaction.activation_energy:
                        self.pending_reactions.append((i, j, reaction))

    def step(self, action: int = ActionType.NOOP):
        """Advance simulation by one step (may use substeps for fast particles)"""
        # apply agent action (once per step)
        self._apply_action(action)

        # clear pending reactions for this step
        self.pending_reactions.clear()

        # substep loop for physics and collision
        sub_dt = self.dt/self.substeps
        for _ in range(self.substeps):
            self._physics_substep(sub_dt)

        # thermal radiation (once per step, after all substeps)
        self._apply_thermal_radiation()

        # process pending reactions (create products, remove reactants)
        self._process_reactions()

        self.step_count += 1

    def _physics_substep(self, sub_dt: float):
        """Single physics substep: forces, movement, collision"""
        # apply electromagnetic forces (charge-based attraction/repulsion)
        em_forces = self._calculate_em_forces()

        # apply world forces (gravity, etc.)
        accelerations = self.world.apply_forces(
            self.positions[:self.n_particles],
            self.velocities[:self.n_particles],
            self.masses[:self.n_particles]
        )

        # update velocities and positions
        for i in range(self.n_particles):
            if not self.active[i]:
                continue
            # combine world forces + em forces
            total_accel = accelerations[i] + em_forces[i]/max(self.masses[i], 0.1)
            self.velocities[i] += total_accel*sub_dt
            self.positions[i] += self.velocities[i]*sub_dt

            # heat-based vibration (scaled by substep)
            temp = self.temperatures[i]
            if temp > 0.01:
                vibration = np.random.randn(2)*temp*self.vibration_scale/np.sqrt(self.substeps)
                self.positions[i] += vibration

            # boundary collision
            new_pos, new_vel = self.world.on_boundary_collision(
                self.positions[i], self.velocities[i],
                (0, 0, self.width-1, self.height-1)
            )
            self.positions[i] = new_pos
            self.velocities[i] = new_vel

        # detect and resolve collisions (may queue reactions)
        collisions = self._detect_collisions()
        for i, j in collisions:
            self._resolve_collision(i, j)

    def _calculate_em_forces(self) -> np.ndarray:
        """
        Calculate electromagnetic forces between charged particles.
        Opposite charges attract, same charges repel.
        Force magnitude: F = k * q1 * q2 / r^2
        """
        forces = np.zeros((self.n_particles, 2), dtype=np.float32)

        for i in range(self.n_particles):
            if not self.active[i] or abs(self.charges[i]) < 0.01:
                continue

            for j in range(i+1, self.n_particles):
                if not self.active[j] or abs(self.charges[j]) < 0.01:
                    continue

                diff = self.positions[j] - self.positions[i]
                dist = np.linalg.norm(diff)

                if dist < 0.1 or dist > self.em_force_radius:
                    continue

                # direction from i to j
                direction = diff/dist

                # Coulomb-like force: opposite charges attract (negative product)
                # same charges repel (positive product)
                q_product = self.charges[i]*self.charges[j]
                force_magnitude = self.em_force_strength*q_product/(dist*dist + 0.5)

                # negative force = attraction (pulls toward), positive = repulsion
                force = -force_magnitude*direction

                forces[i] += force
                forces[j] -= force  # Newton's 3rd law

        return forces

    def _apply_thermal_radiation(self):
        """Apply thermal radiation: heat flows from hot to cold particles at distance"""
        radiation_heat = np.zeros(self.n_particles, dtype=np.float32)

        for i in range(self.n_particles):
            if not self.active[i] or self.temperatures[i] < 0.01:
                continue

            temp_i = self.temperatures[i]

            # radiate to environment (ambient loss)
            ambient_loss = temp_i*self.heat_dissipation
            radiation_heat[i] -= ambient_loss

            # radiate to nearby particles
            for j in range(self.n_particles):
                if i == j or not self.active[j]:
                    continue
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < self.thermal_radiation_radius and dist > 0:
                    temp_j = self.temperatures[j]
                    if temp_i > temp_j:
                        # heat flows from i to j
                        falloff = 1.0 - dist/self.thermal_radiation_radius
                        heat_transfer = (temp_i - temp_j)*self.thermal_radiation_rate*falloff
                        radiation_heat[i] -= heat_transfer
                        radiation_heat[j] += heat_transfer

        # apply accumulated radiation
        for i in range(self.n_particles):
            if self.active[i]:
                self.temperatures[i] = max(0, self.temperatures[i] + radiation_heat[i])

    def _process_reactions(self):
        """Process queued reactions: merge reactants into products"""
        processed = set()

        for i, j, reaction in self.pending_reactions:
            # skip if either particle already reacted this step
            if i in processed or j in processed:
                continue
            if not self.active[i] or not self.active[j]:
                continue

            # create product particle at midpoint
            mid_pos = (self.positions[i] + self.positions[j])/2
            avg_vel = (self.velocities[i]*self.masses[i] + self.velocities[j]*self.masses[j])/(self.masses[i] + self.masses[j])

            # combined temperature + reaction heat delta
            combined_temp = (self.temperatures[i] + self.temperatures[j])/2 + reaction.heat_delta

            # spawn product
            product_idx = self.add_typed_particle(
                x=mid_pos[0], y=mid_pos[1],
                type_name=reaction.product,
                vx=avg_vel[0], vy=avg_vel[1],
                temperature=max(0, combined_temp),
            )

            if product_idx >= 0:
                # deactivate reactants
                self.active[i] = False
                self.active[j] = False
                processed.add(i)
                processed.add(j)

    def render(self, scale: int = 1) -> np.ndarray:
        """Render current state to RGB image"""
        h, w = self.height*scale, self.width*scale
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # draw particles as circles
        for i in range(self.n_particles):
            if not self.active[i]:
                continue

            px, py = int(self.positions[i][0]*scale), int(self.positions[i][1]*scale)
            r = int(self.radii[i]*scale)
            color = (np.array(self.colors[i])*255).astype(np.uint8)

            # draw filled circle
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            img[ny, nx] = color

        return img

    def get_state(self) -> dict:
        """Get current simulation state as dict"""
        return {
            'positions': self.positions[:self.n_particles].copy(),
            'velocities': self.velocities[:self.n_particles].copy(),
            'radii': self.radii[:self.n_particles].copy(),
            'masses': self.masses[:self.n_particles].copy(),
            'types': self.types[:self.n_particles].copy(),
            'type_names': self.type_names[:self.n_particles].copy(),
            'temperatures': self.temperatures[:self.n_particles].copy(),
            'thermal_conductivities': self.thermal_conductivities[:self.n_particles].copy(),
            'charges': self.charges[:self.n_particles].copy(),
            'active': self.active[:self.n_particles].copy(),
            'n_particles': self.n_particles,
            'step': self.step_count,
            'world_config': self.world.config.to_dict(),
        }

    def reset(self):
        """Reset simulation"""
        self.positions.fill(0)
        self.velocities.fill(0)
        self.radii.fill(0)
        self.masses.fill(0)
        self.types.fill(0)
        self.type_names = [''] * self.max_particles
        self.temperatures.fill(0)
        self.thermal_conductivities.fill(0)
        self.charges.fill(0)
        self.colors.fill(0)
        self.active.fill(False)
        self.n_particles = 0
        self.agent_idx = -1
        self.step_count = 0
        self.pending_reactions.clear()

    def add_heat(self, idx: int, amount: float):
        """Add heat to a specific particle"""
        if 0 <= idx < self.n_particles and self.active[idx]:
            self.temperatures[idx] = max(0, self.temperatures[idx] + amount)

    def add_heat_at(self, x: float, y: float, radius: float, amount: float):
        """Add heat to all particles within radius of (x, y)"""
        for i in range(self.n_particles):
            if not self.active[i]:
                continue
            dist = np.linalg.norm(self.positions[i] - np.array([x, y]))
            if dist < radius:
                falloff = 1.0 - dist/radius
                self.temperatures[i] = max(0, self.temperatures[i] + amount*falloff)
