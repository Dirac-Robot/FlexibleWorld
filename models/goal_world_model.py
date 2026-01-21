"""
Goal-Conditioned World Model for RL-based simulation control.

Architecture:
    Goal (text) → GoalEncoder → goal_embedding
    State (image) → StateEncoder → state_embedding
    
    [state_embedding, goal_embedding] → PolicyHead → action_distribution
    [state_embedding, action] → DynamicsModel → next_state_embedding
    [next_state_embedding, goal_embedding] → RewardPredictor → reward
    [next_state_embedding] → Decoder → next_frame (optional)

Training:
    - Supervised: Learn from (state, goal, action, reward) demonstrations
    - RL: PPO/DPO style policy optimization with reward signal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .backbone import (
    VisionEncoderWrapper,
    LLMBackboneWrapper,
    create_clip_encoder,
    create_dinov2_encoder,
    create_llm_backbone,
    _create_placeholder_encoder,
    _create_placeholder_llm,
)


@dataclass
class RLOutput:
    """Output from RL forward pass"""
    action_logits: torch.Tensor  # (B, action_dim) or (B, seq, action_dim)
    action_probs: torch.Tensor
    value: Optional[torch.Tensor] = None  # (B,) value estimate
    state_embedding: Optional[torch.Tensor] = None
    goal_embedding: Optional[torch.Tensor] = None
    predicted_reward: Optional[torch.Tensor] = None
    predicted_next_state: Optional[torch.Tensor] = None
    predicted_frame: Optional[torch.Tensor] = None


class GoalEncoder(nn.Module):
    """
    Encode natural language goal to embedding.
    Uses LLM backbone or simple embedding layer.
    """

    def __init__(self, 
                 llm_backbone: Optional[LLMBackboneWrapper] = None,
                 vocab_size: int = 32000,
                 embed_dim: int = 768,
                 hidden_dim: int = 512,
                 use_llm: bool = False):
        super().__init__()
        self.use_llm = use_llm and llm_backbone is not None
        self.embed_dim = embed_dim

        if self.use_llm:
            self.llm = llm_backbone
            self.proj = nn.Linear(llm_backbone.output_dim, embed_dim)
        else:
            # Simple learned embedding for goal types
            # For production, replace with proper tokenizer + transformer
            self.goal_type_embed = nn.Embedding(16, hidden_dim)  # 16 goal types
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, embed_dim),
            )

    def forward(self, goal_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            goal_tokens: (B, seq_len) token ids or (B,) goal type indices
        
        Returns:
            (B, embed_dim) goal embedding
        """
        if self.use_llm:
            # Use LLM to encode
            llm_out = self.llm(goal_tokens)  # (B, seq_len, llm_dim)
            # Pool to single vector (use last token or mean)
            pooled = llm_out.mean(dim=1)  # (B, llm_dim)
            return self.proj(pooled)
        else:
            # Simple embedding lookup
            if goal_tokens.dim() == 1:
                embed = self.goal_type_embed(goal_tokens)  # (B, hidden_dim)
            else:
                # Use first token as goal type
                embed = self.goal_type_embed(goal_tokens[:, 0])
            return self.proj(embed)


class StateEncoder(nn.Module):
    """
    Encode visual state to embedding.
    Uses vision encoder (CLIP/DINOv2), CNN, or MLP for vector states.
    """

    def __init__(self,
                 vision_encoder: Optional[VisionEncoderWrapper] = None,
                 image_size: int = 64,
                 channels: int = 3,
                 embed_dim: int = 768,
                 use_pretrained: bool = False,
                 state_dim: int = None):  # For vector states
        super().__init__()
        self.use_pretrained = use_pretrained and vision_encoder is not None
        self.embed_dim = embed_dim
        self.state_dim = state_dim  # If set, use MLP for vector input

        if state_dim is not None:
            # MLP for vector state (positions + velocities)
            self.encoder = None
            self.proj = nn.Sequential(
                nn.Linear(state_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        elif self.use_pretrained:
            self.encoder = vision_encoder
            self.proj = nn.Linear(vision_encoder.output_dim, embed_dim)
        else:
            # Simple CNN encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, 32, 4, 2, 1),
                nn.SiLU(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.SiLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.SiLU(),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.SiLU(),
                nn.Flatten(),
            )
            # Calculate flattened size
            feat_size = (image_size//16)**2*256
            self.proj = nn.Linear(feat_size, embed_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (B, C, H, W), (B, T, C, H, W), or (B, state_dim) for vector
        
        Returns:
            (B, embed_dim) or (B, T, embed_dim) state embedding
        """
        # Vector state mode
        if self.state_dim is not None:
            return self.proj(states)
        
        # Image state mode
        images = states
        has_time = images.dim() == 5
        if has_time:
            B, T, C, H, W = images.shape
            images = images.reshape(B*T, C, H, W)

        if self.use_pretrained:
            features = self.encoder(images)
        else:
            features = self.encoder(images)
        
        embed = self.proj(features)

        if has_time:
            embed = embed.reshape(B, T, -1)

        return embed


class PolicyHead(nn.Module):
    """
    Policy head: (state, goal) → action distribution.
    Outputs discrete action probabilities.
    """

    def __init__(self,
                 state_dim: int = 768,
                 goal_dim: int = 768,
                 hidden_dim: int = 512,
                 action_dim: int = 8,
                 action_param_dim: int = 6):
        """
        Args:
            action_dim: Number of discrete action types
            action_param_dim: Dimension of continuous action parameters
                (target, x, y, value, radius, property_type)
        """
        super().__init__()
        self.action_dim = action_dim
        self.action_param_dim = action_param_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim+goal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Discrete action type head
        self.action_type_head = nn.Linear(hidden_dim, action_dim)

        # Continuous parameter heads (predicted per action)
        self.param_mean_head = nn.Linear(hidden_dim, action_param_dim)
        self.param_logstd_head = nn.Linear(hidden_dim, action_param_dim)

    def forward(self, state_embed: torch.Tensor, 
                goal_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_embed: (B, state_dim)
            goal_embed: (B, goal_dim)
        
        Returns:
            Dict with action_type_logits, param_mean, param_logstd
        """
        combined = torch.cat([state_embed, goal_embed], dim=-1)
        hidden = self.net(combined)

        action_type_logits = self.action_type_head(hidden)
        param_mean = self.param_mean_head(hidden)
        param_logstd = self.param_logstd_head(hidden).clamp(-5, 2)

        return {
            'action_type_logits': action_type_logits,
            'action_type_probs': F.softmax(action_type_logits, dim=-1),
            'param_mean': param_mean,
            'param_std': param_logstd.exp(),
        }

    def sample_action(self, state_embed: torch.Tensor, 
                      goal_embed: torch.Tensor,
                      deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Sample action from policy"""
        out = self.forward(state_embed, goal_embed)

        # Sample discrete action type
        if deterministic:
            action_type = out['action_type_probs'].argmax(dim=-1)
        else:
            action_type = torch.distributions.Categorical(
                probs=out['action_type_probs']
            ).sample()

        # Sample continuous parameters
        if deterministic:
            params = out['param_mean']
        else:
            params = torch.distributions.Normal(
                out['param_mean'], out['param_std']
            ).sample()

        # Combine into action vector
        action = torch.cat([
            action_type.unsqueeze(-1).float(),
            params
        ], dim=-1)

        # Compute log prob
        type_log_prob = torch.distributions.Categorical(
            probs=out['action_type_probs']
        ).log_prob(action_type)
        
        param_log_prob = torch.distributions.Normal(
            out['param_mean'], out['param_std']
        ).log_prob(params).sum(dim=-1)

        log_prob = type_log_prob + param_log_prob

        return action, log_prob, out


class ValueHead(nn.Module):
    """Value function: (state, goal) → value estimate"""

    def __init__(self, state_dim: int = 768, goal_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+goal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_embed: torch.Tensor, goal_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state_embed, goal_embed], dim=-1)
        return self.net(combined).squeeze(-1)


class DynamicsModel(nn.Module):
    """Predict next state: (state, action) → next_state"""

    def __init__(self, state_dim: int = 768, action_dim: int = 7, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state_embed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state_embed, action], dim=-1)
        return self.net(combined)


class RewardPredictor(nn.Module):
    """Predict reward: (state, goal) → reward"""

    def __init__(self, state_dim: int = 768, goal_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim+goal_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_embed: torch.Tensor, goal_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state_embed, goal_embed], dim=-1)
        return self.net(combined).squeeze(-1)


class FrameDecoder(nn.Module):
    """Decode state embedding to image"""

    def __init__(self, state_dim: int = 768, image_size: int = 64, channels: int = 3):
        super().__init__()
        self.image_size = image_size
        self.channels = channels

        init_size = image_size//16
        self.proj = nn.Linear(state_dim, 256*init_size*init_size)
        self.init_size = init_size

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_embed: torch.Tensor) -> torch.Tensor:
        B = state_embed.shape[0]
        x = self.proj(state_embed)
        x = x.reshape(B, 256, self.init_size, self.init_size)
        return self.decoder(x)


class GoalConditionedWorldModel(nn.Module):
    """
    Complete Goal-Conditioned World Model for RL.

    Usage:
        model = GoalConditionedWorldModel(...)
        
        # Training: compute losses
        outputs = model(states, goals, actions, rewards)
        policy_loss = model.compute_policy_loss(outputs, actions)
        dynamics_loss = model.compute_dynamics_loss(outputs, next_states)
        reward_loss = model.compute_reward_loss(outputs, rewards)
        
        # Inference: sample actions
        action, log_prob, info = model.sample_action(state, goal)
        
        # Imagination: predict future
        imagined_states, imagined_rewards = model.imagine(state, goal, horizon=10)
    """

    def __init__(
        self,
        state_encoder: Optional[StateEncoder] = None,
        goal_encoder: Optional[GoalEncoder] = None,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        action_dim: int = 8,
        action_param_dim: int = 6,
        image_size: int = 64,
        channels: int = 3,
        use_value_head: bool = True,
        use_dynamics: bool = True,
        use_reward_predictor: bool = True,
        use_frame_decoder: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_param_dim = action_param_dim
        self.total_action_dim = 1 + action_param_dim  # type + params

        # Encoders
        self.state_encoder = state_encoder or StateEncoder(
            image_size=image_size, channels=channels, embed_dim=embed_dim
        )
        self.goal_encoder = goal_encoder or GoalEncoder(embed_dim=embed_dim)

        # Policy
        self.policy = PolicyHead(
            state_dim=embed_dim, goal_dim=embed_dim,
            hidden_dim=hidden_dim, action_dim=action_dim,
            action_param_dim=action_param_dim
        )

        # Optional components
        self.value_head = ValueHead(embed_dim, embed_dim, hidden_dim//2) if use_value_head else None
        self.dynamics = DynamicsModel(embed_dim, self.total_action_dim, hidden_dim) if use_dynamics else None
        self.reward_predictor = RewardPredictor(embed_dim, embed_dim, hidden_dim//2) if use_reward_predictor else None
        self.frame_decoder = FrameDecoder(embed_dim, image_size, channels) if use_frame_decoder else None

    def encode_state(self, images: torch.Tensor) -> torch.Tensor:
        """Encode visual state"""
        return self.state_encoder(images)

    def encode_goal(self, goal_tokens: torch.Tensor) -> torch.Tensor:
        """Encode goal text/tokens"""
        return self.goal_encoder(goal_tokens)

    def forward(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        next_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            states: (B, C, H, W) current state images
            goals: (B,) or (B, seq_len) goal tokens
            actions: (B, action_dim) optional actions for dynamics
            next_states: (B, C, H, W) optional next states for training
        """
        # Encode
        state_embed = self.encode_state(states)
        goal_embed = self.encode_goal(goals)

        # Policy
        policy_out = self.policy(state_embed, goal_embed)

        output = {
            'state_embed': state_embed,
            'goal_embed': goal_embed,
            **policy_out,
        }

        # Value
        if self.value_head is not None:
            output['value'] = self.value_head(state_embed, goal_embed)

        # Dynamics (if actions provided)
        if self.dynamics is not None and actions is not None:
            next_state_pred = self.dynamics(state_embed, actions)
            output['next_state_pred'] = next_state_pred

            # Predicted reward
            if self.reward_predictor is not None:
                output['reward_pred'] = self.reward_predictor(next_state_pred, goal_embed)

            # Decode predicted frame
            if self.frame_decoder is not None:
                output['next_frame_pred'] = self.frame_decoder(next_state_pred)

        return output

    def sample_action(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Sample action from policy.
        
        Returns:
            action: (B, action_dim) sampled action
            log_prob: (B,) log probability
            info: dict with policy outputs including value
        """
        state_embed = self.encode_state(states)
        goal_embed = self.encode_goal(goals)
        action, log_prob, info = self.policy.sample_action(state_embed, goal_embed, deterministic)
        
        # Add value estimate
        if self.value_head is not None:
            info['value'] = self.value_head(state_embed, goal_embed)
        else:
            info['value'] = torch.zeros(states.shape[0], device=states.device)
        
        info['state_embed'] = state_embed
        info['goal_embed'] = goal_embed
        
        return action, log_prob, info

    def imagine(
        self,
        initial_state: torch.Tensor,
        goal: torch.Tensor,
        horizon: int = 10,
        deterministic: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Imagination rollout using learned dynamics.
        
        Returns:
            states: list of state embeddings
            actions: list of actions
            rewards: list of predicted rewards
        """
        if self.dynamics is None:
            raise RuntimeError("Dynamics model required for imagination")

        state_embed = self.encode_state(initial_state)
        goal_embed = self.encode_goal(goal)

        states = [state_embed]
        actions = []
        rewards = []

        for _ in range(horizon):
            # Sample action
            action, _, _ = self.policy.sample_action(state_embed, goal_embed, deterministic)
            actions.append(action)

            # Predict next state
            state_embed = self.dynamics(state_embed, action)
            states.append(state_embed)

            # Predict reward
            if self.reward_predictor is not None:
                reward = self.reward_predictor(state_embed, goal_embed)
                rewards.append(reward)

        return states, actions, rewards

    def compute_policy_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised policy loss (behavior cloning).
        """
        # Cross entropy for action type
        target_type = target_actions[:, 0].long()
        type_loss = F.cross_entropy(outputs['action_type_logits'], target_type)

        # MSE for continuous parameters
        target_params = target_actions[:, 1:]
        param_loss = F.mse_loss(outputs['param_mean'], target_params)

        return type_loss + param_loss

    def compute_dynamics_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dynamics prediction loss"""
        target_embed = self.encode_state(next_states)
        return F.mse_loss(outputs['next_state_pred'], target_embed)

    def compute_reward_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward prediction loss"""
        return F.mse_loss(outputs['reward_pred'], target_rewards)

    def compute_ppo_loss(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        clip_ratio: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss for RL training.
        """
        state_embed = self.encode_state(states)
        goal_embed = self.encode_goal(goals)

        # Get current policy
        policy_out = self.policy(state_embed, goal_embed)

        # Action log prob
        action_type = actions[:, 0].long()
        action_params = actions[:, 1:]

        type_log_prob = torch.distributions.Categorical(
            probs=policy_out['action_type_probs']
        ).log_prob(action_type)

        param_log_prob = torch.distributions.Normal(
            policy_out['param_mean'], policy_out['param_std']
        ).log_prob(action_params).sum(dim=-1)

        new_log_probs = type_log_prob + param_log_prob

        # PPO clipped objective
        ratio = (new_log_probs - old_log_probs).exp()
        clipped_ratio = ratio.clamp(1-clip_ratio, 1+clip_ratio)
        policy_loss = -torch.min(ratio*advantages, clipped_ratio*advantages).mean()

        # Value loss
        value_loss = torch.tensor(0.0, device=states.device)
        if self.value_head is not None:
            values = self.value_head(state_embed, goal_embed)
            value_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy = torch.distributions.Categorical(
            probs=policy_out['action_type_probs']
        ).entropy().mean()

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': policy_loss + 0.5*value_loss - 0.01*entropy,
        }


def create_goal_world_model(config) -> GoalConditionedWorldModel:
    """Factory function to create GoalConditionedWorldModel from config."""

    # State encoder
    state_encoder = None
    state_dim = config.model.get('state_dim', None)
    
    if state_dim is not None:
        # Vector state mode (for particle simulation)
        state_encoder = StateEncoder(
            state_dim=state_dim,
            embed_dim=config.model.embed_dim,
        )
    else:
        # Vision encoder mode
        vision_type = config.model.vision.get('type', 'cnn')

        if vision_type == 'clip':
            vision_enc = create_clip_encoder(
                model_name=config.model.vision.get('name', 'openai/clip-vit-base-patch32'),
                freeze=config.model.vision.get('freeze', True),
            )
            state_encoder = StateEncoder(
                vision_encoder=vision_enc,
                embed_dim=config.model.embed_dim,
                use_pretrained=True,
            )
        elif vision_type == 'dinov2':
            vision_enc = create_dinov2_encoder(
                model_name=config.model.vision.get('name', 'dinov2_vitb14'),
                freeze=config.model.vision.get('freeze', True),
            )
            state_encoder = StateEncoder(
                vision_encoder=vision_enc,
                embed_dim=config.model.embed_dim,
                use_pretrained=True,
            )
        # else: use default CNN encoder (state_encoder=None)

    # Goal encoder (with optional LLM)
    goal_encoder = None
    goal_type = config.model.goal.get('type', 'embed')

    if goal_type == 'llm':
        llm = create_llm_backbone(
            model_name=config.model.goal.get('llm_name', 'meta-llama/Llama-2-7b-hf'),
            freeze=config.model.goal.get('freeze', True),
            load_in_8bit=config.model.goal.get('load_in_8bit', False),
        )
        goal_encoder = GoalEncoder(
            llm_backbone=llm,
            embed_dim=config.model.embed_dim,
            use_llm=True,
        )
    # else: use default embedding encoder (goal_encoder=None)

    return GoalConditionedWorldModel(
        state_encoder=state_encoder,
        goal_encoder=goal_encoder,
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        action_dim=config.sim.action.n_action_types,
        action_param_dim=config.sim.action.action_dim - 1,
        image_size=config.model.image_size,
        channels=config.model.channels,
        use_value_head=config.rl.get('use_value_head', True),
        use_dynamics=config.rl.get('use_dynamics', True),
        use_reward_predictor=config.rl.get('use_reward_predictor', True),
        use_frame_decoder=config.rl.get('use_frame_decoder', False),
    )
