"""
Goal-Conditioned Environment for Agent-Style World Model
Model learns: state + target → action
Supports natural language goals via TextGoal proxy
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from PIL import Image
import re

from simulator.action_operator import ActionOperator, Action, ActionType, PropertyType


class GoalType(Enum):
    """Types of goals that can be specified"""
    POSITION = 'position'           # move particles to positions
    CLUSTER = 'cluster'             # group particles together
    SCATTER = 'scatter'             # spread particles apart
    HEAT = 'heat'                   # raise temperature
    COOL = 'cool'                   # lower temperature
    EXPLODE = 'explode'             # create explosion
    LEFT = 'left'                   # move left
    RIGHT = 'right'                 # move right
    UP = 'up'                       # move up
    DOWN = 'down'                   # move down
    CENTER = 'center'               # move to center
    CORNER = 'corner'               # move to corner


@dataclass
class TextGoal:
    """
    Natural language goal with proxy target representation.
    
    Usage:
        goal = TextGoal.from_text("move all particles to the left")
        target = goal.to_target(env)  # generates proxy target
    """
    text: str
    goal_type: GoalType
    magnitude: float = 1.0  # strength/amount
    direction: Optional[Tuple[float, float]] = None  # for directional goals

    @classmethod
    def from_text(cls, text: str) -> 'TextGoal':
        """Parse natural language goal to TextGoal"""
        text = text.lower().strip()
        
        # Direction patterns
        if any(w in text for w in ['left', '왼쪽']):
            return cls(text, GoalType.LEFT, direction=(-1, 0))
        if any(w in text for w in ['right', '오른쪽']):
            return cls(text, GoalType.RIGHT, direction=(1, 0))
        if any(w in text for w in ['up', '위', '올려']):
            return cls(text, GoalType.UP, direction=(0, -1))
        if any(w in text for w in ['down', '아래', '내려']):
            return cls(text, GoalType.DOWN, direction=(0, 1))
        if any(w in text for w in ['center', '중앙', '가운데']):
            return cls(text, GoalType.CENTER)
        if any(w in text for w in ['corner', '모서리', '구석']):
            return cls(text, GoalType.CORNER)
        
        # Action patterns
        if any(w in text for w in ['cluster', 'group', '모아', '뭉쳐', '모으']):
            return cls(text, GoalType.CLUSTER)
        if any(w in text for w in ['scatter', 'spread', '흩어', '퍼뜨려', '분산']):
            return cls(text, GoalType.SCATTER)
        if any(w in text for w in ['heat', 'hot', 'warm', '가열', '뜨겁', '데워']):
            return cls(text, GoalType.HEAT)
        if any(w in text for w in ['cool', 'cold', '냉각', '차갑', '식혀']):
            return cls(text, GoalType.COOL)
        if any(w in text for w in ['explode', 'explosion', 'boom', '폭발', '터뜨려']):
            return cls(text, GoalType.EXPLODE)
        
        # Default
        return cls(text, GoalType.POSITION)

    def to_target(self, env: 'GoalConditionedEnv') -> Dict[str, np.ndarray]:
        """Convert goal to proxy target state"""
        n = env.operator.sim.n_particles
        if n == 0:
            return {'target_positions': np.array([]), 'goal_text': self.text}
        
        current_pos = env.operator.sim.positions[:n].copy()
        w, h = env.width, env.height
        
        if self.goal_type == GoalType.LEFT:
            target_pos = current_pos.copy()
            target_pos[:, 0] = np.clip(target_pos[:, 0] - 20, 10, w-10)
        
        elif self.goal_type == GoalType.RIGHT:
            target_pos = current_pos.copy()
            target_pos[:, 0] = np.clip(target_pos[:, 0] + 20, 10, w-10)
        
        elif self.goal_type == GoalType.UP:
            target_pos = current_pos.copy()
            target_pos[:, 1] = np.clip(target_pos[:, 1] - 20, 10, h-10)
        
        elif self.goal_type == GoalType.DOWN:
            target_pos = current_pos.copy()
            target_pos[:, 1] = np.clip(target_pos[:, 1] + 20, 10, h-10)
        
        elif self.goal_type == GoalType.CENTER:
            center = np.array([w/2, h/2])
            target_pos = np.tile(center, (n, 1)) + np.random.randn(n, 2)*3
        
        elif self.goal_type == GoalType.CORNER:
            corners = np.array([[10, 10], [w-10, 10], [10, h-10], [w-10, h-10]])
            corner = corners[np.random.randint(4)]
            target_pos = np.tile(corner, (n, 1)) + np.random.randn(n, 2)*5
        
        elif self.goal_type == GoalType.CLUSTER:
            centroid = current_pos.mean(axis=0)
            target_pos = centroid + np.random.randn(n, 2)*5
        
        elif self.goal_type == GoalType.SCATTER:
            # Move particles away from center
            centroid = current_pos.mean(axis=0)
            directions = current_pos - centroid
            directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
            target_pos = current_pos + directions*20
            target_pos = np.clip(target_pos, 10, [w-10, h-10])
        
        elif self.goal_type in [GoalType.HEAT, GoalType.EXPLODE]:
            # Scatter outward (explosion effect)
            centroid = current_pos.mean(axis=0)
            directions = current_pos - centroid
            directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8)
            target_pos = current_pos + directions*25
            target_pos = np.clip(target_pos, 5, [w-5, h-5])
        
        elif self.goal_type == GoalType.COOL:
            # Particles slow down, stay in place
            target_pos = current_pos.copy()
        
        else:  # POSITION - random
            target_pos = np.random.uniform([10, 10], [w-10, h-10], size=(n, 2))
        
        # Store and render
        env.target_positions = target_pos.astype(np.float32)
        env.target_state = env._render_target_positions()
        
        return {
            'target_image': env.target_state,
            'target_positions': env.target_positions.copy(),
            'goal_text': self.text,
            'goal_type': self.goal_type.value,
        }

    def get_embedding(self, dim: int = 32) -> np.ndarray:
        """
        Get a simple embedding for the goal.
        For proper NL understanding, replace with real text encoder.
        """
        # Simple one-hot + magnitude encoding
        emb = np.zeros(dim, dtype=np.float32)
        emb[self.goal_type.value.__hash__() % (dim-2)] = 1.0
        emb[-2] = self.magnitude
        if self.direction:
            emb[-1] = np.arctan2(self.direction[1], self.direction[0]) / np.pi
        return emb


class GoalConditionedEnv:
    """
    Goal-conditioned environment for agent-style world model.
    
    The model receives:
    - current state observation
    - target state observation
    
    And must predict:
    - action to reach target
    
    Usage:
        env = GoalConditionedEnv()
        state = env.reset()
        target = env.sample_target()
        
        # Model predicts action
        action = model.predict(state, target)
        
        # Execute and get reward
        next_state, reward, done, info = env.step(action)
        # reward = how close next_state is to target
    """

    def __init__(self, 
                 width: int = 64, 
                 height: int = 64,
                 max_steps: int = 100,
                 max_particles: int = 30):
        
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.max_particles = max_particles
        
        self.operator = ActionOperator(width=width, height=height)
        self.current_step = 0
        
        # Current target
        self.target_state: Optional[np.ndarray] = None
        self.target_positions: Optional[np.ndarray] = None
        
        # Action bounds
        self.action_dim = 7
        self.action_low = np.array([0, -1, 0, 0, -5, 1, 0], dtype=np.float32)
        self.action_high = np.array([7, max_particles, width, height, 5, 30, 7], dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Reset and return initial observation dict"""
        if seed is not None:
            np.random.seed(seed)
        
        self.operator.reset()
        self.current_step = 0
        self.target_state = None
        self.target_positions = None
        
        # Create initial particles
        self._setup_initial_particles()
        
        return self.get_observation()

    def _setup_initial_particles(self, n_particles: int = 8):
        """Create initial random particles"""
        for _ in range(n_particles):
            x = np.random.uniform(10, self.width-10)
            y = np.random.uniform(10, self.height-10)
            charge = np.random.uniform(-1, 1)
            self.operator.execute(Action(
                ActionType.ADD_PARTICLE, x=x, y=y, value=charge, radius=2.5
            ))

    def sample_target(self, target_type: str = 'position') -> Dict[str, np.ndarray]:
        """
        Sample a target state.
        
        Args:
            target_type: 'position' (move particles), 'configuration' (full layout)
        
        Returns:
            Target observation dict
        """
        if target_type == 'position':
            return self._sample_position_target()
        elif target_type == 'configuration':
            return self._sample_configuration_target()
        else:
            return self._sample_position_target()

    def _sample_position_target(self) -> Dict[str, np.ndarray]:
        """Sample random target positions for existing particles"""
        n = self.operator.sim.n_particles
        
        # Random target positions
        self.target_positions = np.random.uniform(
            [10, 10], [self.width-10, self.height-10],
            size=(n, 2)
        ).astype(np.float32)
        
        # Render target state
        self.target_state = self._render_target_positions()
        
        return {
            'target_image': self.target_state,
            'target_positions': self.target_positions.copy(),
        }

    def _sample_configuration_target(self) -> Dict[str, np.ndarray]:
        """Sample a completely new configuration as target"""
        # Store current state
        current_positions = self.operator.sim.positions[:self.operator.sim.n_particles].copy()
        current_velocities = self.operator.sim.velocities[:self.operator.sim.n_particles].copy()
        current_charges = self.operator.sim.charges[:self.operator.sim.n_particles].copy()
        
        # Run simulation forward randomly to get target
        for _ in range(np.random.randint(10, 30)):
            # Random action
            if np.random.random() < 0.3:
                x, y = np.random.uniform(10, self.width-10), np.random.uniform(10, self.height-10)
                self.operator.execute(Action(ActionType.APPLY_HEAT, x=x, y=y, value=1.0, radius=15))
            self.operator.execute(Action(ActionType.STEP))
        
        # Save target
        self.target_positions = self.operator.sim.positions[:self.operator.sim.n_particles].copy()
        self.target_state = self.operator.get_observation()
        
        # Restore current state
        n = len(current_positions)
        self.operator.sim.positions[:n] = current_positions
        self.operator.sim.velocities[:n] = current_velocities
        self.operator.sim.charges[:n] = current_charges
        
        return {
            'target_image': self.target_state,
            'target_positions': self.target_positions.copy(),
        }

    def _render_target_positions(self) -> np.ndarray:
        """Render target positions as image"""
        # Temporarily move particles to target positions
        n = min(len(self.target_positions), self.operator.sim.n_particles)
        original = self.operator.sim.positions[:n].copy()
        
        self.operator.sim.positions[:n] = self.target_positions[:n]
        target_img = self.operator.get_observation()
        self.operator.sim.positions[:n] = original
        
        return target_img

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute action and compute reward based on target proximity.
        
        Returns:
            (observation, reward, done, info)
        """
        # Parse and execute action
        action = np.clip(action, self.action_low, self.action_high)
        parsed = self._parse_action(action)
        success = self.operator.execute(parsed)
        
        if parsed.action_type != ActionType.STEP:
            self.operator.execute(Action(ActionType.STEP))
        
        self.current_step += 1
        
        # Get observation
        obs = self.get_observation()
        
        # Compute reward (distance to target)
        reward = self._compute_target_reward()
        
        # Done condition
        done = self.current_step >= self.max_steps or reward > -0.1
        
        info = {
            'step': self.current_step,
            'distance': -reward,
            'success': success,
        }
        
        return obs, reward, done, info

    def _parse_action(self, action: np.ndarray) -> Action:
        """Convert action vector to Action"""
        return Action(
            action_type=ActionType(int(np.clip(action[0], 0, 7))),
            target=int(action[1]),
            x=float(action[2]),
            y=float(action[3]),
            value=float(action[4]),
            radius=float(action[5]),
            property_type=PropertyType(int(np.clip(action[6], 0, 7)))
        )

    def _compute_target_reward(self) -> float:
        """
        Compute reward based on distance to target.
        Reward = negative average distance (closer = higher reward)
        """
        if self.target_positions is None:
            return 0.0
        
        n = min(len(self.target_positions), self.operator.sim.n_particles)
        if n == 0:
            return 0.0
        
        current = self.operator.sim.positions[:n]
        target = self.target_positions[:n]
        
        distances = np.linalg.norm(current - target, axis=1)
        avg_distance = np.mean(distances)
        
        # Normalize by diagonal
        max_dist = np.sqrt(self.width**2 + self.height**2)
        normalized = avg_distance / max_dist
        
        return -normalized  # negative distance as reward

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation including target"""
        state_img = self.operator.get_observation().astype(np.float32) / 255.0
        
        obs = {
            'state': state_img,
            'positions': self.operator.sim.positions[:self.operator.sim.n_particles].copy(),
            'n_particles': self.operator.sim.n_particles,
        }
        
        if self.target_state is not None:
            obs['target'] = self.target_state.astype(np.float32) / 255.0
        if self.target_positions is not None:
            obs['target_positions'] = self.target_positions.copy()
        
        return obs

    def render(self, scale: int = 4, show_target: bool = True) -> np.ndarray:
        """Render current state and target side by side"""
        current = self.operator.get_observation(scale=scale)
        
        if show_target and self.target_state is not None:
            # Scale target similarly
            target_pil = Image.fromarray(self.target_state)
            target_pil = target_pil.resize(
                (self.width*scale, self.height*scale), 
                Image.Resampling.NEAREST
            )
            target = np.array(target_pil)
            
            # Concatenate side by side
            combined = np.concatenate([current, target], axis=1)
            return combined
        
        return current


class GoalDataCollector:
    """Collect (state, target, action, next_state) data for training"""

    def __init__(self, env: GoalConditionedEnv):
        self.env = env
        self.data: List[Dict] = []

    def collect_episode(self, n_steps: int = 50) -> List[Dict]:
        """Collect one episode of goal-conditioned data"""
        episode_data = []
        
        state = self.env.reset()
        target = self.env.sample_target()
        
        for _ in range(n_steps):
            # Random action (replace with expert/policy for better data)
            action = np.random.uniform(
                self.env.action_low, 
                self.env.action_high
            ).astype(np.float32)
            
            next_state, reward, done, info = self.env.step(action)
            
            episode_data.append({
                'state': state['state'].copy(),
                'target': target['target_image'].copy() if 'target_image' in target else target.get('target_positions'),
                'action': action.copy(),
                'next_state': next_state['state'].copy(),
                'reward': reward,
            })
            
            state = next_state
            
            if done:
                break
        
        self.data.extend(episode_data)
        return episode_data

    def collect_dataset(self, n_episodes: int = 100, n_steps: int = 50) -> Dict:
        """Collect full dataset"""
        for _ in range(n_episodes):
            self.collect_episode(n_steps)
        
        return {
            'states': np.array([d['state'] for d in self.data]),
            'targets': np.array([d['target'] for d in self.data]),
            'actions': np.array([d['action'] for d in self.data]),
            'next_states': np.array([d['next_state'] for d in self.data]),
            'rewards': np.array([d['reward'] for d in self.data]),
        }

    def save(self, path: str):
        """Save dataset"""
        dataset = self.collect_dataset(0, 0)  # just convert existing
        dataset = {k: np.array([d[k] for d in self.data]) for k in self.data[0].keys()}
        np.savez(path, **dataset)
