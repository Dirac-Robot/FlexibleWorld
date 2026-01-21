"""
Meta Action System - Hierarchical Action Composition

기본 액션들을 조합해서 고차원 액션(메타 액션)을 정의.
예: 입자들을 조합해서 강체, 막, 모터 등을 생성.

Usage:
    registry = MetaActionRegistry()
    registry.register("rigid_body", RigidBodyMacro())
    
    # 실행
    primitives = registry.compile("rigid_body", params={'n_particles': 10, 'center': (32, 32)})
    for action in primitives:
        operator.execute(action)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from enum import IntEnum, auto

from simulator.action_operator import Action, ActionType, PropertyType


class MetaActionType(IntEnum):
    """High-level meta action types"""
    # Structure creation
    CREATE_RIGID_BODY = 100
    CREATE_SOFT_BODY = 101
    CREATE_MEMBRANE = 102
    CREATE_CHAIN = 103
    CREATE_GRID = 104
    
    # Complex behaviors
    CREATE_MOTOR = 110
    CREATE_OSCILLATOR = 111
    CREATE_PUMP = 112
    
    # Chemical structures
    CREATE_MOLECULE = 120
    CREATE_BOND = 121
    BREAK_BOND = 122
    
    # Biological
    CREATE_CELL = 130
    DIVIDE_CELL = 131
    
    # Custom (user-defined)
    CUSTOM = 200


@dataclass
class MetaActionSpec:
    """Specification for a meta action"""
    name: str
    meta_type: MetaActionType
    description: str = ""
    param_spec: Dict[str, Any] = field(default_factory=dict)  # {param_name: (type, default, range)}
    

class MetaAction(ABC):
    """
    Base class for meta actions.
    
    Subclass this to define new meta actions that compile
    into sequences of primitive actions.
    """
    
    @property
    @abstractmethod
    def spec(self) -> MetaActionSpec:
        """Return specification for this meta action"""
        pass
    
    @abstractmethod
    def compile(self, params: Dict[str, Any]) -> List[Action]:
        """
        Compile this meta action into primitive actions.
        
        Args:
            params: Parameters for this meta action
            
        Returns:
            List of primitive Action objects to execute
        """
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill default parameters"""
        validated = {}
        for name, (ptype, default, prange) in self.spec.param_spec.items():
            if name in params:
                validated[name] = ptype(params[name])
            else:
                validated[name] = default
        return validated


class RigidBodyMacro(MetaAction):
    """Create a rigid body from particles with strong bonds"""
    
    @property
    def spec(self) -> MetaActionSpec:
        return MetaActionSpec(
            name="rigid_body",
            meta_type=MetaActionType.CREATE_RIGID_BODY,
            description="Create rigid body from bonded particles",
            param_spec={
                'center_x': (float, 32.0, (0, 64)),
                'center_y': (float, 32.0, (0, 64)),
                'n_particles': (int, 6, (3, 50)),
                'radius': (float, 8.0, (2, 30)),
                'bond_strength': (float, 10.0, (1, 100)),
                'shape': (str, 'circle', ['circle', 'line', 'triangle', 'square']),
            }
        )
    
    def compile(self, params: Dict[str, Any]) -> List[Action]:
        p = self.validate_params(params)
        actions = []
        
        # Generate particle positions based on shape
        positions = self._generate_shape_positions(
            p['center_x'], p['center_y'],
            p['n_particles'], p['radius'], p['shape']
        )
        
        # Add particles
        for x, y in positions:
            actions.append(Action(
                action_type=ActionType.ADD_PARTICLE,
                x=x, y=y,
                value=1.0,  # mass
            ))
        
        # Create strong bonds (via attraction)
        # Bond strength encoded as repeated attractions to make them "stick"
        for i, (x, y) in enumerate(positions):
            actions.append(Action(
                action_type=ActionType.APPLY_ATTRACTION,
                x=x, y=y,
                value=p['bond_strength'],
                radius=p['radius']*0.5,
            ))
        
        return actions
    
    def _generate_shape_positions(
        self, cx: float, cy: float, n: int, r: float, shape: str
    ) -> List[Tuple[float, float]]:
        positions = []
        
        if shape == 'circle':
            for i in range(n):
                angle = 2*np.pi*i/n
                x = cx+r*np.cos(angle)
                y = cy+r*np.sin(angle)
                positions.append((x, y))
                
        elif shape == 'line':
            for i in range(n):
                x = cx-r+2*r*i/(n-1) if n > 1 else cx
                positions.append((x, cy))
                
        elif shape == 'triangle':
            # Distribute on triangle edges
            for i in range(n):
                t = i/n
                if t < 1/3:
                    # Bottom edge
                    s = t*3
                    x = cx-r+s*r
                    y = cy+r*0.866
                elif t < 2/3:
                    # Right edge
                    s = (t-1/3)*3
                    x = cx+r*(1-s)*0.5
                    y = cy+r*0.866-s*r*0.866*2
                else:
                    # Left edge
                    s = (t-2/3)*3
                    x = cx-r*0.5+s*r*0.5
                    y = cy-r*0.866+s*r*0.866*2
                positions.append((x, y))
                
        elif shape == 'square':
            side = int(np.ceil(np.sqrt(n)))
            spacing = 2*r/(side-1) if side > 1 else 0
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= n:
                        break
                    x = cx-r+i*spacing
                    y = cy-r+j*spacing
                    positions.append((x, y))
                    idx += 1
        
        return positions


class SoftBodyMacro(MetaAction):
    """Create a soft/deformable body"""
    
    @property
    def spec(self) -> MetaActionSpec:
        return MetaActionSpec(
            name="soft_body",
            meta_type=MetaActionType.CREATE_SOFT_BODY,
            description="Create deformable soft body",
            param_spec={
                'center_x': (float, 32.0, (0, 64)),
                'center_y': (float, 32.0, (0, 64)),
                'n_particles': (int, 12, (5, 100)),
                'radius': (float, 10.0, (3, 30)),
                'stiffness': (float, 2.0, (0.1, 10)),
            }
        )
    
    def compile(self, params: Dict[str, Any]) -> List[Action]:
        p = self.validate_params(params)
        actions = []
        
        # Fill circle with particles
        positions = []
        sqrt_n = int(np.ceil(np.sqrt(p['n_particles'])))
        spacing = 2*p['radius']/(sqrt_n+1)
        
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                x = p['center_x']-p['radius']+spacing*(i+1)
                y = p['center_y']-p['radius']+spacing*(j+1)
                # Only include if inside circle
                if (x-p['center_x'])**2+(y-p['center_y'])**2 <= p['radius']**2:
                    positions.append((x, y))
                    if len(positions) >= p['n_particles']:
                        break
            if len(positions) >= p['n_particles']:
                break
        
        # Add particles
        for x, y in positions:
            actions.append(Action(
                action_type=ActionType.ADD_PARTICLE,
                x=x, y=y,
                value=0.5,  # lighter mass for soft body
            ))
        
        # Soft bonds (weaker attraction)
        actions.append(Action(
            action_type=ActionType.APPLY_ATTRACTION,
            x=p['center_x'], y=p['center_y'],
            value=p['stiffness'],
            radius=p['radius']*1.5,
        ))
        
        return actions


class MembraneMacro(MetaAction):
    """Create a cell-like membrane (hollow ring)"""
    
    @property
    def spec(self) -> MetaActionSpec:
        return MetaActionSpec(
            name="membrane",
            meta_type=MetaActionType.CREATE_MEMBRANE,
            description="Create hollow membrane structure",
            param_spec={
                'center_x': (float, 32.0, (0, 64)),
                'center_y': (float, 32.0, (0, 64)),
                'n_particles': (int, 16, (6, 50)),
                'outer_radius': (float, 12.0, (5, 30)),
                'thickness': (float, 2.0, (1, 5)),
                'bond_strength': (float, 5.0, (1, 20)),
            }
        )
    
    def compile(self, params: Dict[str, Any]) -> List[Action]:
        p = self.validate_params(params)
        actions = []
        
        # Create ring of particles
        for i in range(p['n_particles']):
            angle = 2*np.pi*i/p['n_particles']
            x = p['center_x']+p['outer_radius']*np.cos(angle)
            y = p['center_y']+p['outer_radius']*np.sin(angle)
            
            actions.append(Action(
                action_type=ActionType.ADD_PARTICLE,
                x=x, y=y,
                value=1.0,
            ))
        
        # Create tangential bonds (along membrane)
        for i in range(p['n_particles']):
            angle = 2*np.pi*i/p['n_particles']
            x = p['center_x']+p['outer_radius']*np.cos(angle)
            y = p['center_y']+p['outer_radius']*np.sin(angle)
            
            actions.append(Action(
                action_type=ActionType.APPLY_ATTRACTION,
                x=x, y=y,
                value=p['bond_strength'],
                radius=p['thickness']*2,
            ))
        
        return actions


class ChainMacro(MetaAction):
    """Create a chain/polymer structure"""
    
    @property
    def spec(self) -> MetaActionSpec:
        return MetaActionSpec(
            name="chain",
            meta_type=MetaActionType.CREATE_CHAIN,
            description="Create chain/polymer of linked particles",
            param_spec={
                'start_x': (float, 16.0, (0, 64)),
                'start_y': (float, 32.0, (0, 64)),
                'end_x': (float, 48.0, (0, 64)),
                'end_y': (float, 32.0, (0, 64)),
                'n_particles': (int, 8, (2, 30)),
                'bond_strength': (float, 3.0, (0.5, 20)),
            }
        )
    
    def compile(self, params: Dict[str, Any]) -> List[Action]:
        p = self.validate_params(params)
        actions = []
        
        # Linear interpolation
        for i in range(p['n_particles']):
            t = i/(p['n_particles']-1) if p['n_particles'] > 1 else 0
            x = p['start_x']+t*(p['end_x']-p['start_x'])
            y = p['start_y']+t*(p['end_y']-p['start_y'])
            
            actions.append(Action(
                action_type=ActionType.ADD_PARTICLE,
                x=x, y=y,
                value=0.8,
            ))
            
            # Local bond
            actions.append(Action(
                action_type=ActionType.APPLY_ATTRACTION,
                x=x, y=y,
                value=p['bond_strength'],
                radius=3.0,
            ))
        
        return actions


class MetaActionRegistry:
    """
    Registry for meta actions.
    
    Allows registration of new meta actions and compilation to primitives.
    """
    
    def __init__(self):
        self._registry: Dict[str, MetaAction] = {}
        self._learned: Dict[str, List[Action]] = {}  # For learned sequences
        
        # Register built-in macros
        self.register(RigidBodyMacro())
        self.register(SoftBodyMacro())
        self.register(MembraneMacro())
        self.register(ChainMacro())
    
    def register(self, macro: MetaAction) -> None:
        """Register a meta action"""
        self._registry[macro.spec.name] = macro
    
    def register_sequence(self, name: str, actions: List[Action]) -> None:
        """Register a learned action sequence as new meta action"""
        self._learned[name] = actions
    
    def list_available(self) -> List[str]:
        """List all available meta actions"""
        return list(self._registry.keys())+list(self._learned.keys())
    
    def get_spec(self, name: str) -> Optional[MetaActionSpec]:
        """Get specification for a meta action"""
        if name in self._registry:
            return self._registry[name].spec
        return None
    
    def compile(self, name: str, params: Dict[str, Any] = None) -> List[Action]:
        """
        Compile a meta action into primitive actions.
        
        Args:
            name: Meta action name
            params: Parameters for the meta action
            
        Returns:
            List of primitive actions
        """
        params = params or {}
        
        if name in self._registry:
            return self._registry[name].compile(params)
        elif name in self._learned:
            return self._learned[name].copy()
        else:
            raise ValueError(f"Unknown meta action: {name}")
    
    def execute(self, operator, name: str, params: Dict[str, Any] = None) -> None:
        """
        Compile and execute a meta action.
        
        Args:
            operator: ActionOperator to execute on
            name: Meta action name
            params: Parameters
        """
        actions = self.compile(name, params)
        for action in actions:
            operator.execute(action)


# Convenience function
def create_registry() -> MetaActionRegistry:
    """Create a new meta action registry with default macros"""
    return MetaActionRegistry()
