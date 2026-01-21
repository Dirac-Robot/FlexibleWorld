"""
LLM Agent for Goal-Conditioned Particle Simulation

Two modes:
1. External LLM: Call external LLM API (e.g., GPT-4) for action generation
2. Learned Policy: Use trained GoalConditionedWorldModel for action generation

LLM receives state + goal → outputs action → environment evaluates goal achievement
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
import base64
from io import BytesIO
from PIL import Image

import torch

from simulator.action_operator import Action, ActionType, PropertyType
from simulator.goal_env import GoalConditionedEnv, TextGoal, GoalType

TEMPLATE_DIR = Path(__file__).parent / 'templates'


def load_template(name: str) -> str:
    """Load a template file from the templates directory."""
    template_path = TEMPLATE_DIR / f'{name}.txt'
    return template_path.read_text(encoding='utf-8')


@dataclass
class LLMResponse:
    """Parsed LLM response"""
    action: Optional[Action] = None
    reasoning: str = ""
    raw_text: str = ""
    success: bool = False
    log_prob: Optional[float] = None
    value: Optional[float] = None


class LLMAgent:
    """
    LLM-based agent for goal-conditioned simulation control.
    
    Usage:
        agent = LLMAgent(llm_fn=your_llm_function)
        
        env = GoalConditionedEnv()
        state = env.reset()
        
        # Natural language goal
        goal = "입자들을 왼쪽으로 이동시켜"
        
        # LLM decides action
        response = agent.act(state, goal)
        
        # Execute and evaluate
        next_state, reward, done, info = env.step(response.action.to_vector())
        
        # Check goal achievement
        achieved = agent.evaluate_goal(next_state, goal)
    """

    def __init__(self, 
                 llm_fn: Optional[Callable[[str, Optional[str]], str]] = None,
                 include_image: bool = True):
        """
        Args:
            llm_fn: Function(prompt, image_base64) -> response_text
                    If None, uses mock LLM for testing
            include_image: Whether to include state image in prompt
        """
        self.llm_fn = llm_fn or self._mock_llm
        self.include_image = include_image
        self.action_history: List[LLMResponse] = []
        self._action_prompt_template = load_template('action_prompt')

    def act(self, state: Dict, goal: str, env: Optional[GoalConditionedEnv] = None) -> LLMResponse:
        """
        Get action from LLM given state and goal.
        
        Args:
            state: Current observation dict from environment
            goal: Natural language goal description
            env: Optional environment for context
        
        Returns:
            LLMResponse with parsed action
        """
        # Build prompt
        prompt = self._build_prompt(state, goal)
        
        # Get image if available
        image_b64 = None
        if self.include_image and 'state' in state:
            image_b64 = self._encode_image(state['state'])
        
        # Call LLM
        raw_response = self.llm_fn(prompt, image_b64)
        
        # Parse response
        response = self._parse_response(raw_response)
        
        self.action_history.append(response)
        return response

    def _build_prompt(self, state: Dict, goal: str) -> str:
        """Build prompt for LLM"""
        n_particles = state.get('n_particles', 0)
        return self._action_prompt_template.format(n_particles=n_particles, goal=goal)

    def _encode_image(self, img_array: np.ndarray) -> str:
        """Encode image to base64"""
        if img_array.max() <= 1.0:
            img_array = (img_array*255).astype(np.uint8)
        
        pil_img = Image.fromarray(img_array)
        buffer = BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def _parse_response(self, raw_text: str) -> LLMResponse:
        """Parse LLM response to extract action"""
        response = LLMResponse(raw_text=raw_text)
        
        try:
            # Try to extract JSON
            text = raw_text.strip()
            
            # Find JSON block
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            
            data = json.loads(text)
            
            response.reasoning = data.get('reasoning', '')
            
            action_data = data.get('action', {})
            action_type_str = action_data.get('type', 'NOOP')
            
            # Map string to ActionType
            type_map = {
                'NOOP': ActionType.NOOP,
                'ADD_PARTICLE': ActionType.ADD_PARTICLE,
                'SET_PROPERTY': ActionType.SET_PROPERTY,
                'APPLY_HEAT': ActionType.APPLY_HEAT,
                'APPLY_FORCE': ActionType.APPLY_FORCE,
                'APPLY_ATTRACTION': ActionType.APPLY_ATTRACTION,
                'APPLY_REPULSION': ActionType.APPLY_REPULSION,
                'STEP': ActionType.STEP,
            }
            
            action_type = type_map.get(action_type_str, ActionType.NOOP)
            
            response.action = Action(
                action_type=action_type,
                x=float(action_data.get('x', 32)),
                y=float(action_data.get('y', 32)),
                value=float(action_data.get('value', 1.0)),
                radius=float(action_data.get('radius', 15.0)),
            )
            response.success = True
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            response.reasoning = f"Parse error: {e}"
            response.action = Action(ActionType.NOOP)
            response.success = False
        
        return response

    def evaluate_goal(self, state: Dict, goal: str, 
                      target: Optional[Dict] = None) -> Tuple[bool, float]:
        """
        Evaluate if the goal was achieved.
        
        Returns:
            (achieved: bool, score: float)
        """
        # Parse goal to get target
        text_goal = TextGoal.from_text(goal)
        
        # Simple heuristics based on goal type
        if target is not None and 'target_positions' in target:
            current = state.get('positions', np.array([]))
            target_pos = target['target_positions']
            
            if len(current) > 0 and len(target_pos) > 0:
                n = min(len(current), len(target_pos))
                distances = np.linalg.norm(current[:n] - target_pos[:n], axis=1)
                avg_dist = np.mean(distances)
                score = max(0, 1 - avg_dist/50)  # normalize
                achieved = avg_dist < 10  # threshold
                return achieved, score
        
        return False, 0.0

    def _mock_llm(self, prompt: str, image_b64: Optional[str] = None) -> str:
        """Mock LLM for testing - returns simple heuristic actions"""
        # Extract goal from prompt
        if 'left' in prompt.lower() or '왼쪽' in prompt:
            return json.dumps({
                "reasoning": "To move particles left, apply rightward force",
                "action": {"type": "APPLY_FORCE", "x": 60, "y": 32, "value": 2.0, "radius": 50}
            })
        elif 'right' in prompt.lower() or '오른쪽' in prompt:
            return json.dumps({
                "reasoning": "To move particles right, apply leftward attraction",
                "action": {"type": "APPLY_ATTRACTION", "x": 60, "y": 32, "value": 2.0, "radius": 50}
            })
        elif 'cluster' in prompt.lower() or '모아' in prompt:
            return json.dumps({
                "reasoning": "To cluster, create attraction at center",
                "action": {"type": "APPLY_ATTRACTION", "x": 32, "y": 32, "value": 3.0, "radius": 40}
            })
        elif 'explode' in prompt.lower() or '폭발' in prompt:
            return json.dumps({
                "reasoning": "To explode, add heat at center",
                "action": {"type": "APPLY_HEAT", "x": 32, "y": 32, "value": 5.0, "radius": 20}
            })
        else:
            return json.dumps({
                "reasoning": "Default: apply small force",
                "action": {"type": "APPLY_FORCE", "x": 32, "y": 32, "value": 1.0, "radius": 30}
            })


def run_episode(agent: LLMAgent, 
                env: GoalConditionedEnv, 
                goal: str, 
                max_steps: int = 20) -> Dict:
    """
    Run a full episode with LLM agent.
    
    Returns:
        Episode results dict
    """
    state = env.reset()
    text_goal = TextGoal.from_text(goal)
    target = text_goal.to_target(env)
    
    total_reward = 0
    actions_taken = []
    
    for step in range(max_steps):
        # LLM decides action
        response = agent.act(state, goal, env)
        
        if response.action is None:
            continue
        
        actions_taken.append({
            'step': step,
            'action_type': response.action.action_type.name,
            'reasoning': response.reasoning,
        })
        
        # Execute action
        action_vec = response.action.to_vector()
        next_state, reward, done, info = env.step(action_vec)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # Final evaluation
    achieved, score = agent.evaluate_goal(state, goal, target)
    
    return {
        'goal': goal,
        'achieved': achieved,
        'score': score,
        'total_reward': total_reward,
        'steps': len(actions_taken),
        'actions': actions_taken,
    }


class LearnedPolicyAgent:
    """
    Agent using a trained GoalConditionedWorldModel as policy.
    
    Usage:
        from models import GoalConditionedWorldModel, create_goal_world_model
        
        # Load trained model
        model = create_goal_world_model(config)
        model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])
        
        agent = LearnedPolicyAgent(model)
        
        env = GoalConditionedEnv()
        state = env.reset()
        
        response = agent.act(state, "move particles left")
        next_state, reward, done, info = env.step(response.action.to_vector())
    """

    def __init__(self, 
                 model: 'GoalConditionedWorldModel',
                 device: torch.device = None,
                 deterministic: bool = False):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        self.deterministic = deterministic
        self.action_history: List[LLMResponse] = []

    @classmethod
    def from_checkpoint(cls, 
                        checkpoint_path: str,
                        config = None,
                        device: torch.device = None) -> 'LearnedPolicyAgent':
        """Load agent from checkpoint"""
        from models import create_goal_world_model

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if config is None:
            from ato.adict import ADict
            config = ADict(checkpoint.get('config', {}))

        model = create_goal_world_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])

        return cls(model, device)

    def act(self, state: Dict, goal: str, env: Optional[GoalConditionedEnv] = None) -> LLMResponse:
        """
        Get action from learned policy.
        
        Args:
            state: Current observation dict from environment
            goal: Natural language goal description
            env: Optional environment for context
        """
        # Prepare state tensor
        state_img = state.get('state', state.get('image'))
        if isinstance(state_img, np.ndarray):
            state_tensor = torch.from_numpy(state_img).float()
            if state_tensor.dim() == 3 and state_tensor.shape[-1] == 3:
                state_tensor = state_tensor.permute(2, 0, 1)
            if state_tensor.max() > 1.0:
                state_tensor = state_tensor / 255.0
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
        else:
            state_tensor = state_img.unsqueeze(0).to(self.device)

        # Parse goal to type index
        text_goal = TextGoal.from_text(goal)
        goal_idx = text_goal.goal_type.value.__hash__() % 16
        goal_tensor = torch.tensor([goal_idx], dtype=torch.long, device=self.device)

        # Get action from model
        with torch.no_grad():
            action, log_prob, info = self.model.sample_action(
                state_tensor, goal_tensor, deterministic=self.deterministic
            )

            value = None
            if self.model.value_head is not None:
                value = self.model.value_head(
                    info['state_embed'] if 'state_embed' in info else self.model.encode_state(state_tensor),
                    info['goal_embed'] if 'goal_embed' in info else self.model.encode_goal(goal_tensor),
                ).item()

        # Convert to Action object
        action_np = action[0].cpu().numpy()
        parsed_action = self._parse_action_vector(action_np)

        response = LLMResponse(
            action=parsed_action,
            reasoning=f"Learned policy for goal: {goal}",
            raw_text=f"action={action_np.tolist()}",
            success=True,
            log_prob=log_prob[0].item(),
            value=value,
        )

        self.action_history.append(response)
        return response

    def _parse_action_vector(self, action_vec: np.ndarray) -> Action:
        """Convert action vector to Action object"""
        action_type = ActionType(int(np.clip(action_vec[0], 0, 7)))

        return Action(
            action_type=action_type,
            target=int(action_vec[1]) if len(action_vec) > 1 else -1,
            x=float(action_vec[2]) if len(action_vec) > 2 else 32.0,
            y=float(action_vec[3]) if len(action_vec) > 3 else 32.0,
            value=float(action_vec[4]) if len(action_vec) > 4 else 1.0,
            radius=float(action_vec[5]) if len(action_vec) > 5 else 15.0,
            property_type=PropertyType(int(np.clip(action_vec[6], 0, 7))) if len(action_vec) > 6 else PropertyType.POSITION_X,
        )

    def evaluate_goal(self, state: Dict, goal: str,
                      target: Optional[Dict] = None) -> Tuple[bool, float]:
        """Evaluate if goal was achieved (same as LLMAgent)"""
        text_goal = TextGoal.from_text(goal)

        if target is not None and 'target_positions' in target:
            current = state.get('positions', np.array([]))
            target_pos = target['target_positions']

            if len(current) > 0 and len(target_pos) > 0:
                n = min(len(current), len(target_pos))
                distances = np.linalg.norm(current[:n] - target_pos[:n], axis=1)
                avg_dist = np.mean(distances)
                score = max(0, 1 - avg_dist/50)
                achieved = avg_dist < 10
                return achieved, score

        return False, 0.0

    def imagine_trajectory(self, state: Dict, goal: str, 
                           horizon: int = 10) -> List[Dict]:
        """
        Imagine future trajectory using learned dynamics.
        
        Returns list of predicted states and rewards.
        """
        if self.model.dynamics is None:
            raise RuntimeError("Model dynamics required for imagination")

        # Prepare inputs
        state_img = state.get('state', state.get('image'))
        if isinstance(state_img, np.ndarray):
            state_tensor = torch.from_numpy(state_img).float()
            if state_tensor.dim() == 3 and state_tensor.shape[-1] == 3:
                state_tensor = state_tensor.permute(2, 0, 1)
            if state_tensor.max() > 1.0:
                state_tensor = state_tensor / 255.0
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
        else:
            state_tensor = state_img.unsqueeze(0).to(self.device)

        text_goal = TextGoal.from_text(goal)
        goal_idx = text_goal.goal_type.value.__hash__() % 16
        goal_tensor = torch.tensor([goal_idx], dtype=torch.long, device=self.device)

        # Imagination rollout
        with torch.no_grad():
            states, actions, rewards = self.model.imagine(
                state_tensor, goal_tensor, horizon, deterministic=True
            )

        trajectory = []
        for i, (s, a, r) in enumerate(zip(states[1:], actions, rewards)):
            trajectory.append({
                'step': i,
                'state_embedding': s[0].cpu().numpy(),
                'action': a[0].cpu().numpy(),
                'predicted_reward': r[0].item() if r.dim() > 0 else r.item(),
            })

        return trajectory


def run_episode_learned(agent: LearnedPolicyAgent,
                        env: GoalConditionedEnv,
                        goal: str,
                        max_steps: int = 20) -> Dict:
    """Run episode with learned policy agent."""
    state = env.reset()
    text_goal = TextGoal.from_text(goal)
    target = text_goal.to_target(env)

    total_reward = 0
    actions_taken = []
    log_probs = []
    values = []

    for step in range(max_steps):
        response = agent.act(state, goal, env)

        if response.action is None:
            continue

        actions_taken.append({
            'step': step,
            'action_type': response.action.action_type.name,
            'action_vec': response.action.to_vector().tolist(),
        })

        if response.log_prob is not None:
            log_probs.append(response.log_prob)
        if response.value is not None:
            values.append(response.value)

        action_vec = response.action.to_vector()
        next_state, reward, done, info = env.step(action_vec)

        total_reward += reward
        state = next_state

        if done:
            break

    achieved, score = agent.evaluate_goal(state, goal, target)

    return {
        'goal': goal,
        'achieved': achieved,
        'score': score,
        'total_reward': total_reward,
        'steps': len(actions_taken),
        'actions': actions_taken,
        'log_probs': log_probs,
        'values': values,
    }
