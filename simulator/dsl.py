"""
Meta Action DSL - Text-based Domain Specific Language for Action Composition

모델(LLM/VLM)이 텍스트로 메타 액션을 생성하면 파싱해서 실행.

Grammar:
    CREATE <type> [AT (<x>, <y>)] [WITH <param>=<value> ...]
    BOND <source> TO <target> [WITH strength=<value>]
    MOVE <target> TO (<x>, <y>) [WITH speed=<value>]
    APPLY <force_type> AT (<x>, <y>) [WITH value=<v> radius=<r>]
    SEQUENCE <name> { <commands> }
    RUN <sequence_name> [WITH <overrides>]

Examples:
    CREATE rigid_body AT (32, 32) WITH n=6 shape=circle radius=10
    CREATE membrane AT (32, 32) WITH n=16 outer=12
    CREATE chain FROM (10, 32) TO (54, 32) WITH n=8
    APPLY force AT (32, 32) WITH value=5 radius=20
    
    SEQUENCE my_cell {
        CREATE membrane AT (32, 32) WITH n=20
        CREATE soft_body AT (32, 32) WITH n=8 radius=6
    }
    RUN my_cell

Usage:
    dsl = MetaDSL(registry)
    actions = dsl.parse("CREATE rigid_body AT (32, 32) WITH n=6")
    for action in actions:
        operator.execute(action)
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import IntEnum, auto

from simulator.action_operator import Action, ActionType
from simulator.meta_action import MetaActionRegistry, create_registry


class DSLCommandType(IntEnum):
    """DSL command types"""
    CREATE = 1
    BOND = 2
    MOVE = 3
    APPLY = 4
    SEQUENCE = 5
    RUN = 6
    STEP = 7
    WAIT = 8


@dataclass
class DSLCommand:
    """Parsed DSL command"""
    command_type: DSLCommandType
    target_type: str = ""
    position: Optional[Tuple[float, float]] = None
    end_position: Optional[Tuple[float, float]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    body: List['DSLCommand'] = field(default_factory=list)  # For SEQUENCE


class DSLParseError(Exception):
    """DSL parsing error"""
    pass


class MetaDSL:
    """
    Text-based DSL parser and compiler for meta actions.
    
    Converts text commands into executable Action sequences.
    """
    
    def __init__(self, registry: MetaActionRegistry = None):
        self.registry = registry or create_registry()
        self.sequences: Dict[str, List[DSLCommand]] = {}
        
        # Regex patterns
        self._coord_pattern = r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
        self._param_pattern = r'(\w+)\s*=\s*([^\s,}]+)'
    
    def parse(self, text: str) -> List[Action]:
        """
        Parse DSL text and return primitive actions.
        
        Args:
            text: DSL command(s) as text
            
        Returns:
            List of primitive Action objects
        """
        commands = self._parse_text(text)
        actions = []
        for cmd in commands:
            actions.extend(self._compile_command(cmd))
        return actions
    
    def parse_multi(self, text: str) -> List[Action]:
        """Parse multiple commands (newline separated)"""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        all_actions = []
        for line in lines:
            if line.startswith('#'):  # Comment
                continue
            all_actions.extend(self.parse(line))
        return all_actions
    
    def _parse_text(self, text: str) -> List[DSLCommand]:
        """Parse text into DSLCommand objects"""
        text = text.strip()
        if not text:
            return []
        
        commands = []
        
        # Check for SEQUENCE definition
        seq_match = re.match(r'SEQUENCE\s+(\w+)\s*\{([^}]*)\}', text, re.IGNORECASE | re.DOTALL)
        if seq_match:
            name = seq_match.group(1)
            body_text = seq_match.group(2)
            body_commands = []
            for line in body_text.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    body_commands.extend(self._parse_text(line))
            cmd = DSLCommand(
                command_type=DSLCommandType.SEQUENCE,
                target_type=name,
                body=body_commands,
            )
            self.sequences[name] = body_commands
            commands.append(cmd)
            return commands
        
        # Parse single command
        parts = text.split()
        if not parts:
            return []
        
        cmd_word = parts[0].upper()
        
        if cmd_word == 'CREATE':
            commands.append(self._parse_create(text))
        elif cmd_word == 'BOND':
            commands.append(self._parse_bond(text))
        elif cmd_word == 'MOVE':
            commands.append(self._parse_move(text))
        elif cmd_word == 'APPLY':
            commands.append(self._parse_apply(text))
        elif cmd_word == 'RUN':
            commands.append(self._parse_run(text))
        elif cmd_word == 'STEP':
            commands.append(DSLCommand(command_type=DSLCommandType.STEP, params={'n': 1}))
        elif cmd_word == 'WAIT':
            n = int(parts[1]) if len(parts) > 1 else 10
            commands.append(DSLCommand(command_type=DSLCommandType.WAIT, params={'n': n}))
        else:
            raise DSLParseError(f"Unknown command: {cmd_word}")
        
        return commands
    
    def _parse_create(self, text: str) -> DSLCommand:
        """Parse CREATE command"""
        # CREATE <type> [AT (x, y)] [FROM (x1, y1) TO (x2, y2)] [WITH params]
        
        # Extract type
        match = re.match(r'CREATE\s+(\w+)', text, re.IGNORECASE)
        if not match:
            raise DSLParseError(f"Invalid CREATE syntax: {text}")
        target_type = match.group(1).lower()
        
        # Extract position (AT)
        position = None
        at_match = re.search(r'AT\s*' + self._coord_pattern, text, re.IGNORECASE)
        if at_match:
            position = (float(at_match.group(1)), float(at_match.group(2)))
        
        # Extract FROM...TO for chains
        end_position = None
        from_match = re.search(r'FROM\s*' + self._coord_pattern, text, re.IGNORECASE)
        to_match = re.search(r'TO\s*' + self._coord_pattern, text, re.IGNORECASE)
        if from_match:
            position = (float(from_match.group(1)), float(from_match.group(2)))
        if to_match:
            end_position = (float(to_match.group(1)), float(to_match.group(2)))
        
        # Extract parameters (WITH)
        params = {}
        with_match = re.search(r'WITH\s+(.+)$', text, re.IGNORECASE)
        if with_match:
            param_text = with_match.group(1)
            for m in re.finditer(self._param_pattern, param_text):
                key = m.group(1).lower()
                value = m.group(2)
                # Try to convert to number
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
        
        return DSLCommand(
            command_type=DSLCommandType.CREATE,
            target_type=target_type,
            position=position,
            end_position=end_position,
            params=params,
        )
    
    def _parse_bond(self, text: str) -> DSLCommand:
        """Parse BOND command"""
        # BOND <source> TO <target> [WITH strength=<value>]
        match = re.match(r'BOND\s+(\w+)\s+TO\s+(\w+)', text, re.IGNORECASE)
        if not match:
            raise DSLParseError(f"Invalid BOND syntax: {text}")
        
        params = {'source': match.group(1), 'target': match.group(2)}
        
        with_match = re.search(r'WITH\s+(.+)$', text, re.IGNORECASE)
        if with_match:
            for m in re.finditer(self._param_pattern, with_match.group(1)):
                params[m.group(1).lower()] = float(m.group(2))
        
        return DSLCommand(
            command_type=DSLCommandType.BOND,
            params=params,
        )
    
    def _parse_move(self, text: str) -> DSLCommand:
        """Parse MOVE command"""
        # MOVE <target> TO (x, y) [WITH speed=<value>]
        match = re.match(r'MOVE\s+(\w+)\s+TO\s*' + self._coord_pattern, text, re.IGNORECASE)
        if not match:
            raise DSLParseError(f"Invalid MOVE syntax: {text}")
        
        params = {'target': match.group(1)}
        position = (float(match.group(2)), float(match.group(3)))
        
        with_match = re.search(r'WITH\s+(.+)$', text, re.IGNORECASE)
        if with_match:
            for m in re.finditer(self._param_pattern, with_match.group(1)):
                params[m.group(1).lower()] = float(m.group(2))
        
        return DSLCommand(
            command_type=DSLCommandType.MOVE,
            position=position,
            params=params,
        )
    
    def _parse_apply(self, text: str) -> DSLCommand:
        """Parse APPLY command"""
        # APPLY <force_type> AT (x, y) [WITH value=<v> radius=<r>]
        match = re.match(r'APPLY\s+(\w+)\s+AT\s*' + self._coord_pattern, text, re.IGNORECASE)
        if not match:
            raise DSLParseError(f"Invalid APPLY syntax: {text}")
        
        force_type = match.group(1).lower()
        position = (float(match.group(2)), float(match.group(3)))
        
        params = {'force_type': force_type}
        with_match = re.search(r'WITH\s+(.+)$', text, re.IGNORECASE)
        if with_match:
            for m in re.finditer(self._param_pattern, with_match.group(1)):
                key = m.group(1).lower()
                try:
                    params[key] = float(m.group(2))
                except ValueError:
                    params[key] = m.group(2)
        
        return DSLCommand(
            command_type=DSLCommandType.APPLY,
            target_type=force_type,
            position=position,
            params=params,
        )
    
    def _parse_run(self, text: str) -> DSLCommand:
        """Parse RUN command"""
        # RUN <sequence_name> [WITH overrides]
        match = re.match(r'RUN\s+(\w+)', text, re.IGNORECASE)
        if not match:
            raise DSLParseError(f"Invalid RUN syntax: {text}")
        
        seq_name = match.group(1)
        params = {}
        
        with_match = re.search(r'WITH\s+(.+)$', text, re.IGNORECASE)
        if with_match:
            for m in re.finditer(self._param_pattern, with_match.group(1)):
                params[m.group(1).lower()] = m.group(2)
        
        return DSLCommand(
            command_type=DSLCommandType.RUN,
            target_type=seq_name,
            params=params,
        )
    
    def _compile_command(self, cmd: DSLCommand) -> List[Action]:
        """Compile a DSLCommand into primitive actions"""
        
        if cmd.command_type == DSLCommandType.CREATE:
            return self._compile_create(cmd)
        
        elif cmd.command_type == DSLCommandType.APPLY:
            return self._compile_apply(cmd)
        
        elif cmd.command_type == DSLCommandType.STEP:
            n = cmd.params.get('n', 1)
            return [Action(action_type=ActionType.STEP) for _ in range(n)]
        
        elif cmd.command_type == DSLCommandType.WAIT:
            n = cmd.params.get('n', 10)
            return [Action(action_type=ActionType.STEP) for _ in range(n)]
        
        elif cmd.command_type == DSLCommandType.RUN:
            seq_name = cmd.target_type
            if seq_name in self.sequences:
                actions = []
                for sub_cmd in self.sequences[seq_name]:
                    actions.extend(self._compile_command(sub_cmd))
                return actions
            else:
                raise DSLParseError(f"Unknown sequence: {seq_name}")
        
        elif cmd.command_type == DSLCommandType.SEQUENCE:
            # Just register, don't execute
            return []
        
        elif cmd.command_type == DSLCommandType.BOND:
            # Bonds are handled via attraction
            params = cmd.params
            strength = params.get('strength', 5.0)
            return [Action(
                action_type=ActionType.APPLY_ATTRACTION,
                value=strength,
                radius=5.0,
            )]
        
        elif cmd.command_type == DSLCommandType.MOVE:
            # Move via force
            pos = cmd.position or (32, 32)
            speed = cmd.params.get('speed', 2.0)
            return [Action(
                action_type=ActionType.APPLY_FORCE,
                x=pos[0], y=pos[1],
                value=speed,
                radius=30.0,
            )]
        
        return []
    
    def _compile_create(self, cmd: DSLCommand) -> List[Action]:
        """Compile CREATE command"""
        target_type = cmd.target_type
        
        # Map DSL names to registry names
        type_map = {
            'rigid_body': 'rigid_body',
            'rigidbody': 'rigid_body',
            'rigid': 'rigid_body',
            'soft_body': 'soft_body',
            'softbody': 'soft_body',
            'soft': 'soft_body',
            'membrane': 'membrane',
            'cell': 'membrane',
            'chain': 'chain',
            'polymer': 'chain',
            'particle': 'particle',
        }
        
        registry_name = type_map.get(target_type, target_type)
        
        # Handle single particle
        if registry_name == 'particle':
            pos = cmd.position or (32, 32)
            return [Action(
                action_type=ActionType.ADD_PARTICLE,
                x=pos[0], y=pos[1],
                value=cmd.params.get('mass', 1.0),
            )]
        
        # Build params for registry
        params = dict(cmd.params)
        if cmd.position:
            params['center_x'] = cmd.position[0]
            params['center_y'] = cmd.position[1]
        if cmd.end_position:
            params['end_x'] = cmd.end_position[0]
            params['end_y'] = cmd.end_position[1]
            if cmd.position:
                params['start_x'] = cmd.position[0]
                params['start_y'] = cmd.position[1]
        
        # Param name normalization
        if 'n' in params:
            params['n_particles'] = params.pop('n')
        if 'r' in params:
            params['radius'] = params.pop('r')
        if 'outer' in params:
            params['outer_radius'] = params.pop('outer')
        
        try:
            return self.registry.compile(registry_name, params)
        except ValueError as e:
            raise DSLParseError(f"Cannot compile {target_type}: {e}")
    
    def _compile_apply(self, cmd: DSLCommand) -> List[Action]:
        """Compile APPLY command"""
        force_type = cmd.target_type
        pos = cmd.position or (32, 32)
        value = cmd.params.get('value', 2.0)
        radius = cmd.params.get('radius', 20.0)
        
        force_map = {
            'force': ActionType.APPLY_FORCE,
            'push': ActionType.APPLY_FORCE,
            'attraction': ActionType.APPLY_ATTRACTION,
            'attract': ActionType.APPLY_ATTRACTION,
            'pull': ActionType.APPLY_ATTRACTION,
            'repulsion': ActionType.APPLY_REPULSION,
            'repel': ActionType.APPLY_REPULSION,
            'heat': ActionType.APPLY_HEAT,
            'warm': ActionType.APPLY_HEAT,
        }
        
        action_type = force_map.get(force_type, ActionType.APPLY_FORCE)
        
        return [Action(
            action_type=action_type,
            x=pos[0], y=pos[1],
            value=value,
            radius=radius,
        )]
    
    def to_text(self, actions: List[Action]) -> str:
        """
        Convert primitive actions back to DSL text.
        (For model output interpretation)
        """
        lines = []
        for action in actions:
            if action.action_type == ActionType.ADD_PARTICLE:
                lines.append(f"CREATE particle AT ({action.x:.1f}, {action.y:.1f})")
            elif action.action_type == ActionType.APPLY_FORCE:
                lines.append(f"APPLY force AT ({action.x:.1f}, {action.y:.1f}) WITH value={action.value:.1f} radius={action.radius:.1f}")
            elif action.action_type == ActionType.APPLY_ATTRACTION:
                lines.append(f"APPLY attraction AT ({action.x:.1f}, {action.y:.1f}) WITH value={action.value:.1f} radius={action.radius:.1f}")
            elif action.action_type == ActionType.APPLY_REPULSION:
                lines.append(f"APPLY repulsion AT ({action.x:.1f}, {action.y:.1f}) WITH value={action.value:.1f} radius={action.radius:.1f}")
            elif action.action_type == ActionType.APPLY_HEAT:
                lines.append(f"APPLY heat AT ({action.x:.1f}, {action.y:.1f}) WITH value={action.value:.1f} radius={action.radius:.1f}")
            elif action.action_type == ActionType.STEP:
                lines.append("STEP")
        return '\n'.join(lines)


def create_dsl(registry: MetaActionRegistry = None) -> MetaDSL:
    """Create a new DSL parser"""
    return MetaDSL(registry)
