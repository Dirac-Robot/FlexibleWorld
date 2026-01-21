"""
Action Space for agent interaction with the world
"""
from enum import IntEnum
from dataclasses import dataclass

import numpy as np


class ActionType(IntEnum):
    """Discrete action types for agent"""
    NOOP = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    PUSH = 5       # push nearby particles
    PULL = 6       # pull nearby particles
    SPAWN = 7      # spawn a new particle


# Action to velocity delta mapping
ACTION_DELTAS = {
    ActionType.NOOP: (0.0, 0.0),
    ActionType.MOVE_UP: (0.0, -1.0),
    ActionType.MOVE_DOWN: (0.0, 1.0),
    ActionType.MOVE_LEFT: (-1.0, 0.0),
    ActionType.MOVE_RIGHT: (1.0, 0.0),
}


@dataclass
class ActionSpace:
    """Action space configuration"""
    n_actions: int = 8
    move_speed: float = 2.0
    push_force: float = 3.0
    pull_force: float = 2.0
    interaction_radius: float = 8.0

    @staticmethod
    def sample() -> int:
        """Random action"""
        return np.random.randint(0, 8)

    @staticmethod
    def get_move_delta(action: int, speed: float = 2.0) -> tuple:
        """Get velocity delta for movement actions"""
        if action in ACTION_DELTAS:
            dx, dy = ACTION_DELTAS[ActionType(action)]
            return dx*speed, dy*speed
        return 0.0, 0.0

    @staticmethod
    def is_interaction(action: int) -> bool:
        """Check if action involves particle interaction"""
        return action in (ActionType.PUSH, ActionType.PULL, ActionType.SPAWN)
