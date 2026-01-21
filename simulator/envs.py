"""
Gym-style Environment for Particle Simulation
Enables RL training with the ActionOperator.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np

from simulator.action_operator import ActionOperator, Action, ActionType, PropertyType


class ParticleEnv:
    """
    Gym-style environment for particle simulation.
    
    Observation: RGB image (H, W, 3) or state dict
    Action: 7-dim continuous vector [action_type, target, x, y, value, radius, property_type]
    Reward: Task-dependent (override compute_reward for custom tasks)
    
    Usage:
        env = ParticleEnv()
        obs = env.reset()
        
        for _ in range(100):
            action = env.action_space_sample()  # or from policy
            obs, reward, done, info = env.step(action)
    """

    def __init__(self, 
                 width: int = 64, 
                 height: int = 64,
                 max_steps: int = 200,
                 max_particles: int = 50,
                 obs_type: str = 'image'):  # 'image' or 'state'
        
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.max_particles = max_particles
        self.obs_type = obs_type
        
        self.operator = ActionOperator(width=width, height=height)
        self.current_step = 0
        
        # Action space bounds
        self.action_low = np.array([0, -1, 0, 0, -5, 1, 0], dtype=np.float32)
        self.action_high = np.array([7, max_particles, width, height, 5, 30, 7], dtype=np.float32)
        
        # Observation space
        if obs_type == 'image':
            self.observation_shape = (height, width, 3)
        else:
            # State vector: positions + velocities + charges + temps for max_particles
            self.observation_shape = (max_particles, 6)  # x, y, vx, vy, charge, temp

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and return initial observation"""
        if seed is not None:
            np.random.seed(seed)
        
        self.operator.reset()
        self.current_step = 0
        
        # Optional: add some initial particles
        self._setup_initial_state()
        
        return self._get_obs()

    def _setup_initial_state(self):
        """Override for custom initial setups"""
        # Default: empty world
        pass

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (obs, reward, done, info).
        
        Args:
            action: 7-dim vector [action_type, target, x, y, value, radius, property_type]
        """
        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)
        
        # Convert to Action object
        parsed_action = self._parse_action(action)
        
        # Execute
        success = self.operator.execute(parsed_action)
        
        # Always step simulation if not a STEP action
        if parsed_action.action_type != ActionType.STEP:
            self.operator.execute(Action(ActionType.STEP))
        
        self.current_step += 1
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward (override for custom tasks)
        reward = self.compute_reward(parsed_action, success)
        
        # Check done
        done = self.current_step >= self.max_steps
        
        # Info
        info = {
            'step': self.current_step,
            'n_particles': self.operator.sim.n_particles,
            'action_success': success,
        }
        
        return obs, reward, done, info

    def _parse_action(self, action: np.ndarray) -> Action:
        """Convert action vector to Action object"""
        action_type = ActionType(int(np.clip(action[0], 0, 7)))
        
        return Action(
            action_type=action_type,
            target=int(action[1]),
            x=float(action[2]),
            y=float(action[3]),
            value=float(action[4]),
            radius=float(action[5]),
            property_type=PropertyType(int(np.clip(action[6], 0, 7)))
        )

    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        if self.obs_type == 'image':
            return self.operator.get_observation(scale=1).astype(np.float32) / 255.0
        else:
            return self._get_state_vector()

    def _get_state_vector(self) -> np.ndarray:
        """Get state as fixed-size vector"""
        state = np.zeros(self.observation_shape, dtype=np.float32)
        n = min(self.operator.sim.n_particles, self.max_particles)
        
        if n > 0:
            state[:n, 0] = self.operator.sim.positions[:n, 0] / self.width
            state[:n, 1] = self.operator.sim.positions[:n, 1] / self.height
            state[:n, 2] = self.operator.sim.velocities[:n, 0] / 10.0
            state[:n, 3] = self.operator.sim.velocities[:n, 1] / 10.0
            state[:n, 4] = self.operator.sim.charges[:n] / 3.0
            state[:n, 5] = self.operator.sim.temperatures[:n] / 10.0
        
        return state

    def compute_reward(self, action: Action, success: bool) -> float:
        """
        Compute reward. Override for custom tasks.
        
        Default: small negative reward per step (encourages efficiency)
        """
        reward = -0.01  # step penalty
        
        if not success:
            reward -= 0.1  # failed action penalty
        
        return reward

    def action_space_sample(self) -> np.ndarray:
        """Sample random action"""
        return np.random.uniform(self.action_low, self.action_high).astype(np.float32)

    def render(self, scale: int = 4) -> np.ndarray:
        """Render current state"""
        return self.operator.get_observation(scale=scale)


class ExplosionEnv(ParticleEnv):
    """
    Task: Create an explosion that pushes particles to the edges.
    Reward: Based on average distance from center.
    """

    def _setup_initial_state(self):
        """Start with particles in center"""
        for _ in range(12):
            x = self.width/2 + np.random.uniform(-10, 10)
            y = self.height/2 + np.random.uniform(-10, 10)
            charge = np.random.choice([-1.0, 1.0])
            self.operator.execute(Action(
                ActionType.ADD_PARTICLE,
                x=x, y=y, value=charge, radius=2.0
            ))

    def compute_reward(self, action: Action, success: bool) -> float:
        """Reward = average distance from center"""
        if self.operator.sim.n_particles == 0:
            return 0.0
        
        positions = self.operator.sim.positions[:self.operator.sim.n_particles]
        center = np.array([self.width/2, self.height/2])
        distances = np.linalg.norm(positions - center, axis=1)
        
        return float(np.mean(distances)) / (self.width/2)  # normalize to ~0-1


class ClusterEnv(ParticleEnv):
    """
    Task: Cluster particles together.
    Reward: Negative of average distance between particles.
    """

    def _setup_initial_state(self):
        """Start with scattered particles"""
        for _ in range(15):
            x = np.random.uniform(10, self.width-10)
            y = np.random.uniform(10, self.height-10)
            self.operator.execute(Action(
                ActionType.ADD_PARTICLE,
                x=x, y=y, value=0.5, radius=2.0
            ))

    def compute_reward(self, action: Action, success: bool) -> float:
        """Reward = negative average pairwise distance"""
        n = self.operator.sim.n_particles
        if n < 2:
            return 0.0
        
        positions = self.operator.sim.positions[:n]
        
        # Compute pairwise distances
        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                total_dist += np.linalg.norm(positions[i] - positions[j])
                count += 1
        
        avg_dist = total_dist / count if count > 0 else 0
        
        # Negative distance = closer is better
        return -avg_dist / self.width  # normalize
