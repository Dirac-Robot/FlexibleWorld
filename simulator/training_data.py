"""
LLM Training Data Generator for Goal-Conditioned Simulation
Generates (state, goal, action, reward) pairs for SFT/RLHF training
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

from simulator.goal_env import GoalConditionedEnv, TextGoal, GoalType
from simulator.action_operator import Action, ActionType


@dataclass
class TrainingSample:
    """Single training sample for LLM"""
    # Required fields (no defaults)
    goal_text: str
    goal_type: str
    state_description: str
    action_json: str
    reasoning: str
    
    # Optional fields (with defaults)
    state_image_path: Optional[str] = None
    reward: float = 0.0
    goal_achieved: bool = False
    
    def to_chat_format(self) -> Dict:
        """Convert to chat format for LLM training"""
        user_msg = f"""Current simulation state:
{self.state_description}

Goal: {self.goal_text}

What action should be taken to achieve this goal?"""
        
        assistant_msg = f"""{{"reasoning": "{self.reasoning}", "action": {self.action_json}}}"""
        
        return {
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ],
            "reward": float(self.reward),
            "goal_achieved": bool(self.goal_achieved),
        }
    
    def to_instruction_format(self) -> Dict:
        """Convert to instruction tuning format"""
        return {
            "instruction": f"Goal: {self.goal_text}\nState: {self.state_description}",
            "input": "",
            "output": f'{{"reasoning": "{self.reasoning}", "action": {self.action_json}}}',
            "reward": float(self.reward),
        }


class ExpertPolicy:
    """
    Simple expert policy that knows optimal actions for goals.
    Used to generate training data.
    """
    
    def __init__(self, env: GoalConditionedEnv):
        self.env = env
        self.width = env.width
        self.height = env.height

    def get_action(self, goal: TextGoal) -> Tuple[Action, str]:
        """
        Get expert action for a goal.
        
        Returns:
            (action, reasoning)
        """
        goal_type = goal.goal_type
        
        # Center of mass of particles
        n = self.env.operator.sim.n_particles
        if n > 0:
            positions = self.env.operator.sim.positions[:n]
            centroid = positions.mean(axis=0)
        else:
            centroid = np.array([self.width/2, self.height/2])
        
        if goal_type == GoalType.LEFT:
            # Apply force from right side
            action = Action(
                ActionType.APPLY_FORCE,
                x=self.width - 10,
                y=centroid[1],
                value=2.0,
                radius=self.width
            )
            reasoning = "Apply force from right to push particles left"
        
        elif goal_type == GoalType.RIGHT:
            action = Action(
                ActionType.APPLY_FORCE,
                x=10,
                y=centroid[1],
                value=2.0,
                radius=self.width
            )
            reasoning = "Apply force from left to push particles right"
        
        elif goal_type == GoalType.UP:
            action = Action(
                ActionType.APPLY_FORCE,
                x=centroid[0],
                y=self.height - 10,
                value=2.0,
                radius=self.height
            )
            reasoning = "Apply force from bottom to push particles up"
        
        elif goal_type == GoalType.DOWN:
            action = Action(
                ActionType.APPLY_FORCE,
                x=centroid[0],
                y=10,
                value=2.0,
                radius=self.height
            )
            reasoning = "Apply force from top to push particles down"
        
        elif goal_type == GoalType.CENTER:
            action = Action(
                ActionType.APPLY_ATTRACTION,
                x=self.width/2,
                y=self.height/2,
                value=3.0,
                radius=max(self.width, self.height)
            )
            reasoning = "Create attraction at center to pull particles inward"
        
        elif goal_type == GoalType.CLUSTER:
            action = Action(
                ActionType.APPLY_ATTRACTION,
                x=centroid[0],
                y=centroid[1],
                value=3.0,
                radius=40
            )
            reasoning = "Create attraction at centroid to cluster particles"
        
        elif goal_type == GoalType.SCATTER:
            action = Action(
                ActionType.APPLY_REPULSION,
                x=centroid[0],
                y=centroid[1],
                value=3.0,
                radius=40
            )
            reasoning = "Create repulsion at center to scatter particles"
        
        elif goal_type in [GoalType.EXPLODE, GoalType.HEAT]:
            action = Action(
                ActionType.APPLY_HEAT,
                x=centroid[0],
                y=centroid[1],
                value=5.0,
                radius=25
            )
            reasoning = "Apply heat at center to trigger explosion"
        
        elif goal_type == GoalType.COOL:
            # No direct cooling action, just wait
            action = Action(ActionType.NOOP)
            reasoning = "Allow natural cooling by not adding energy"
        
        else:  # CORNER or unknown
            corners = [(10, 10), (self.width-10, 10), 
                      (10, self.height-10), (self.width-10, self.height-10)]
            corner = corners[np.random.randint(4)]
            action = Action(
                ActionType.APPLY_ATTRACTION,
                x=corner[0],
                y=corner[1],
                value=3.0,
                radius=50
            )
            reasoning = f"Create attraction at corner ({corner[0]}, {corner[1]})"
        
        return action, reasoning


class TrainingDataGenerator:
    """
    Generate training data for LLM fine-tuning.
    
    Usage:
        generator = TrainingDataGenerator(output_dir='./training_data')
        generator.generate(n_episodes=1000)
        generator.save()
    """

    def __init__(self, 
                 env: Optional[GoalConditionedEnv] = None,
                 output_dir: str = './training_data',
                 save_images: bool = False):
        
        self.env = env or GoalConditionedEnv(width=64, height=64)
        self.expert = ExpertPolicy(self.env)
        self.output_dir = Path(output_dir)
        self.save_images = save_images
        
        self.samples: List[TrainingSample] = []
        
        # Goal templates for variety
        self.goal_templates = {
            GoalType.LEFT: [
                "move all particles to the left",
                "push particles leftward",
                "입자들을 왼쪽으로 이동시켜",
                "shift everything left",
            ],
            GoalType.RIGHT: [
                "move particles to the right",
                "push everything rightward", 
                "입자들을 오른쪽으로",
            ],
            GoalType.UP: [
                "move particles upward",
                "push particles to the top",
                "위로 올려",
            ],
            GoalType.DOWN: [
                "move particles down",
                "아래로 내려",
            ],
            GoalType.CENTER: [
                "move all particles to the center",
                "가운데로 모아",
                "gather at center",
            ],
            GoalType.CLUSTER: [
                "cluster the particles together",
                "group all particles",
                "입자들을 뭉쳐",
                "make them stick together",
            ],
            GoalType.SCATTER: [
                "scatter the particles",
                "spread them apart",
                "흩어지게 해",
            ],
            GoalType.EXPLODE: [
                "create an explosion",
                "폭발시켜",
                "make it explode",
                "boom!",
            ],
            GoalType.HEAT: [
                "heat up the particles",
                "가열해",
                "add heat",
            ],
        }

    def _describe_state(self) -> str:
        """Generate text description of current state"""
        n = self.env.operator.sim.n_particles
        if n == 0:
            return "No particles in simulation"
        
        positions = self.env.operator.sim.positions[:n]
        centroid = positions.mean(axis=0)
        spread = positions.std(axis=0).mean()
        
        desc = f"{n} particles, center at ({centroid[0]:.0f}, {centroid[1]:.0f}), spread: {spread:.1f}"
        return desc

    def _action_to_json(self, action: Action) -> str:
        """Convert action to JSON string"""
        return json.dumps({
            "type": action.action_type.name,
            "x": round(float(action.x), 1),
            "y": round(float(action.y), 1),
            "value": round(float(action.value), 1),
            "radius": round(float(action.radius), 1),
        })

    def generate_episode(self, goal_type: Optional[GoalType] = None) -> List[TrainingSample]:
        """Generate one episode of training data"""
        samples = []
        
        # Reset environment
        self.env.reset()
        
        # Pick goal
        if goal_type is None:
            goal_type = np.random.choice(list(self.goal_templates.keys()))
        
        goal_texts = self.goal_templates.get(goal_type, ["achieve the goal"])
        goal_text = np.random.choice(goal_texts)
        
        text_goal = TextGoal.from_text(goal_text)
        target = text_goal.to_target(self.env)
        
        # Get expert action
        action, reasoning = self.expert.get_action(text_goal)
        
        # Create sample before action
        state_desc = self._describe_state()
        
        # Execute action and get reward
        if action.action_type != ActionType.NOOP:
            action_vec = action.to_vector()
            _, reward, _, _ = self.env.step(action_vec)
        else:
            reward = 0.0
        
        # Evaluate goal
        achieved = reward > -0.2  # threshold
        
        sample = TrainingSample(
            goal_text=goal_text,
            goal_type=goal_type.value,
            state_description=state_desc,
            action_json=self._action_to_json(action),
            reasoning=reasoning,
            reward=reward,
            goal_achieved=achieved,
        )
        
        samples.append(sample)
        return samples

    def generate(self, n_episodes: int = 1000, progress: bool = True) -> List[TrainingSample]:
        """Generate full training dataset"""
        self.samples = []
        
        for i in range(n_episodes):
            episode_samples = self.generate_episode()
            self.samples.extend(episode_samples)
            
            if progress and (i+1) % 100 == 0:
                print(f"Generated {i+1}/{n_episodes} episodes")
        
        return self.samples

    def save(self, format: str = 'jsonl'):
        """Save training data"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            # Chat format for fine-tuning
            chat_path = self.output_dir / 'train_chat.jsonl'
            with open(chat_path, 'w') as f:
                for sample in self.samples:
                    f.write(json.dumps(sample.to_chat_format()) + '\n')
            print(f"Saved {len(self.samples)} samples to {chat_path}")
            
            # Instruction format
            inst_path = self.output_dir / 'train_instruction.jsonl'
            with open(inst_path, 'w') as f:
                for sample in self.samples:
                    f.write(json.dumps(sample.to_instruction_format()) + '\n')
            print(f"Saved to {inst_path}")
        
        elif format == 'json':
            path = self.output_dir / 'train_data.json'
            with open(path, 'w') as f:
                json.dump([asdict(s) for s in self.samples], f, indent=2)
            print(f"Saved to {path}")

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        if not self.samples:
            return {}
        
        goal_types = [s.goal_type for s in self.samples]
        rewards = [s.reward for s in self.samples]
        achieved = [s.goal_achieved for s in self.samples]
        
        return {
            'total_samples': len(self.samples),
            'goal_distribution': {g: goal_types.count(g) for g in set(goal_types)},
            'avg_reward': np.mean(rewards),
            'achievement_rate': np.mean(achieved),
        }
