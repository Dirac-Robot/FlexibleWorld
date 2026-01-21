"""
Particle Type Registry - defines particle species and their reactions
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List


@dataclass
class Reaction:
    """Defines a reaction between two particle types"""
    reactant_a: str          # first particle type name
    reactant_b: str          # second particle type name
    product: str             # resulting particle type
    activation_energy: float # minimum collision energy to trigger reaction
    heat_delta: float        # positive=exothermic (release heat), negative=endothermic (absorb heat)


@dataclass
class ParticleType:
    """Definition of a particle species"""
    name: str
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mass: float = 1.0
    radius: float = 2.0
    thermal_conductivity: float = 0.5
    base_temperature: float = 0.0

    # electromagnetic properties
    charge: float = 0.0  # core charge for EM forces

    # valence chemistry properties (for emergent bonding)
    valence: int = 0  # bonding capacity: + = electron donor, - = electron acceptor
    electronegativity: float = 0.5  # 0-1, tendency to attract electrons
    bond_energy: float = 0.0  # energy released when forming bonds (negative = absorbs)
    bond_radius: float = 4.0  # distance at which bonding can occur

    # visual properties
    glow_at_temperature: float = 2.0  # temperature at which particle starts glowing


class ParticleRegistry:
    """Registry of all particle types and their reactions"""

    def __init__(self):
        self.types: Dict[str, ParticleType] = {}
        self.reactions: Dict[Tuple[str, str], Reaction] = {}
        self._register_default_types()
        self._register_default_reactions()

    def _register_default_types(self):
        """Register built-in particle types with valence chemistry"""
        # Hydrogen: valence +1 (electron donor), forms bonds easily
        self.register_type(ParticleType(
            name='hydrogen',
            color=(0.8, 0.8, 1.0),
            mass=0.5,
            radius=1.5,
            thermal_conductivity=0.9,
            valence=1,  # wants to give 1 electron
            electronegativity=0.2,
            bond_energy=2.5,  # releases energy when bonding
        ))

        # Oxygen: valence -2 (electron acceptor)
        self.register_type(ParticleType(
            name='oxygen',
            color=(0.9, 0.3, 0.3),
            mass=1.0,
            radius=2.0,
            thermal_conductivity=0.7,
            valence=-2,  # wants to accept 2 electrons
            electronegativity=0.7,
            bond_energy=1.5,
        ))

        # Water: valence 0 (stable, no bonding)
        self.register_type(ParticleType(
            name='water',
            color=(0.2, 0.5, 0.9),
            mass=1.5,
            radius=2.0,
            thermal_conductivity=0.8,
            valence=0,  # stable, won't bond
            electronegativity=0.5,
            bond_energy=0.0,
        ))

        # Carbon: valence +4 or -4 (flexible)
        self.register_type(ParticleType(
            name='carbon',
            color=(0.2, 0.2, 0.2),
            mass=1.2,
            radius=2.0,
            thermal_conductivity=0.3,
            valence=4,  # can give 4 electrons
            electronegativity=0.5,
            bond_energy=3.0,
        ))

        # Carbon Dioxide: stable
        self.register_type(ParticleType(
            name='carbon_dioxide',
            color=(0.5, 0.5, 0.5),
            mass=2.2,
            radius=2.5,
            thermal_conductivity=0.6,
            valence=0,
            electronegativity=0.5,
            bond_energy=0.0,
        ))

        # Fire: high energy, reactive
        self.register_type(ParticleType(
            name='fire',
            color=(1.0, 0.4, 0.1),
            mass=0.3,
            radius=2.0,
            thermal_conductivity=1.0,
            base_temperature=5.0,
            glow_at_temperature=0.5,
            valence=1,  # reactive, gives energy
            electronegativity=0.1,
            bond_energy=4.0,  # high energy release
        ))

        # Ash: inert
        self.register_type(ParticleType(
            name='ash',
            color=(0.4, 0.4, 0.4),
            mass=0.8,
            radius=1.5,
            thermal_conductivity=0.2,
            valence=0,
            bond_energy=0.0,
        ))

        # Metal: conductive, stable
        self.register_type(ParticleType(
            name='metal',
            color=(0.7, 0.7, 0.8),
            mass=3.0,
            radius=2.5,
            thermal_conductivity=0.95,
            valence=0,  # metallic bonding (special)
            electronegativity=0.3,
            bond_energy=0.5,
        ))

        # Ice: solid water, can melt
        self.register_type(ParticleType(
            name='ice',
            color=(0.7, 0.9, 1.0),
            mass=1.4,
            radius=2.0,
            thermal_conductivity=0.6,
            valence=0,
            bond_energy=-1.0,  # absorbs heat to break bonds (melt)
        ))

        # Steam: gas water
        self.register_type(ParticleType(
            name='steam',
            color=(0.9, 0.9, 0.95),
            mass=0.6,
            radius=2.5,
            thermal_conductivity=0.4,
        ))

        # Agent particle (special)
        self.register_type(ParticleType(
            name='agent',
            color=(0.2, 0.8, 0.2),  # green
            mass=2.0,
            radius=3.0,
            thermal_conductivity=0.5,
        ))

    def _register_default_reactions(self):
        """Register built-in reactions"""
        # Hydrogen + Oxygen -> Water (exothermic, explosive)
        self.register_reaction(Reaction(
            reactant_a='hydrogen',
            reactant_b='oxygen',
            product='water',
            activation_energy=2.0,
            heat_delta=5.0,  # releases heat
        ))

        # Carbon + Oxygen -> Carbon Dioxide (exothermic, burning)
        self.register_reaction(Reaction(
            reactant_a='carbon',
            reactant_b='oxygen',
            product='carbon_dioxide',
            activation_energy=1.5,
            heat_delta=3.0,
        ))

        # Carbon + Fire -> Ash (exothermic)
        self.register_reaction(Reaction(
            reactant_a='carbon',
            reactant_b='fire',
            product='ash',
            activation_energy=0.5,
            heat_delta=2.0,
        ))

        # Ice + Fire -> Water (endothermic, absorbs heat)
        self.register_reaction(Reaction(
            reactant_a='ice',
            reactant_b='fire',
            product='water',
            activation_energy=0.3,
            heat_delta=-2.0,  # absorbs heat
        ))

        # Water (high energy) -> Steam (phase change, endothermic)
        # Note: This is self-reaction, handled differently
        self.register_reaction(Reaction(
            reactant_a='water',
            reactant_b='fire',
            product='steam',
            activation_energy=1.0,
            heat_delta=-1.5,
        ))

    def register_type(self, ptype: ParticleType):
        """Register a new particle type"""
        self.types[ptype.name] = ptype

    def register_reaction(self, reaction: Reaction):
        """Register a reaction (both orderings are stored)"""
        key1 = (reaction.reactant_a, reaction.reactant_b)
        key2 = (reaction.reactant_b, reaction.reactant_a)
        self.reactions[key1] = reaction
        self.reactions[key2] = reaction

    def get_type(self, name: str) -> Optional[ParticleType]:
        """Get particle type by name"""
        return self.types.get(name)

    def get_reaction(self, type_a: str, type_b: str) -> Optional[Reaction]:
        """Get reaction between two particle types, if any"""
        return self.reactions.get((type_a, type_b))

    def list_types(self) -> List[str]:
        """List all registered particle type names"""
        return list(self.types.keys())


# Global registry instance
PARTICLE_REGISTRY = ParticleRegistry()
