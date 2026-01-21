"""
VLM Policy Training for Goal-Conditioned World Model

Uses Vision-Language Model (Qwen2-VL) to directly process:
- Image observation (64x64 RGB)
- Natural language goal

Outputs: Action (type + parameters)

Usage:
    # Single GPU
    python train_vlm.py vlm_full

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 train_vlm.py vlm_full
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from loguru import logger
from PIL import Image

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
os.environ.setdefault('HF_HOME', '/workspace/.cache/huggingface')

from config import scope
from utils import save_checkpoint, load_checkpoint
from simulator.goal_env import GoalConditionedEnv
from simulator.action_operator import ActionType

sys.path.insert(0, str(Path(__file__).parent/'scripts'))
from generate_training_data import GoalDSL, SimState, GOAL_GENERATORS


def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        return {
            'rank': rank,
            'world_size': world_size,
            'local_rank': local_rank,
            'is_main': rank == 0,
            'device': f'cuda:{local_rank}',
        }

    return {
        'rank': 0,
        'world_size': 1,
        'local_rank': 0,
        'is_main': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(dist_info: dict) -> bool:
    return dist_info['is_main']


def reduce_dict(input_dict: dict, dist_info: dict) -> dict:
    """Reduce dictionary values across all processes."""
    if dist_info['world_size'] == 1:
        return input_dict

    with torch.no_grad():
        keys = list(input_dict.keys())
        values = torch.tensor([input_dict[k] for k in keys], device=dist_info['device'])
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist_info['world_size']
        return {k: v.item() for k, v in zip(keys, values)}


NATURAL_COMMANDS = {
    'directional_push': ['모든 입자를 위로 밀어', 'Push up', '아래로', '왼쪽으로', '오른쪽으로'],
    'cluster': ['입자들을 모아', '뭉쳐', 'Cluster', '모아줘'],
    'spread': ['퍼뜨려', '흩어', 'Spread', '분산시켜'],
    'vibrate': ['진동시켜', '열을 가해', 'Vibrate'],
    'heat': ['뜨겁게', '가열해', 'Heat up'],
    'move_to': ['오른쪽으로', '왼쪽으로', '위로', '아래로'],
    'align': ['정렬해', '일렬로', 'Align'],
}

DIRECTION_VECTORS = {
    'up': [0, -1],
    'down': [0, 1],
    'left': [-1, 0],
    'right': [1, 0],
}


class VLMPolicy(nn.Module):
    """
    Vision-Language Model Policy using Qwen2-VL.

    Input: Image (64x64 RGB) + Text Goal
    Output: Action distribution (type + parameters)
    """

    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2-VL-2B-Instruct',
        action_dim: int = 8,
        action_param_dim: int = 6,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        device: str = 'cuda',
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.action_param_dim = action_param_dim
        self.model_name = model_name

        logger.info(f'Loading VLM: {model_name}')

        if 'Qwen2-VL' in model_name:
            self._load_qwen2_vl(model_name, use_lora, lora_rank, lora_alpha)
        else:
            raise ValueError(f'Unsupported VLM: {model_name}')

        hidden_size = self.vlm.config.hidden_size

        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SiLU(),
        ).to(device)

        self.action_type_head = nn.Linear(256, action_dim).to(device)
        self.action_param_head = nn.Linear(256, action_param_dim).to(device)
        self.action_param_logstd = nn.Parameter(torch.zeros(action_param_dim, device=device))

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        ).to(device)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'VLMPolicy trainable parameters: {trainable_params:,}')

    def _load_qwen2_vl(self, model_name: str, use_lora: bool, lora_rank: int, lora_alpha: int):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir='/workspace/.cache/huggingface',
        )

        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir='/workspace/.cache/huggingface',
            device_map=self.device,
        )

        for param in self.vlm.parameters():
            param.requires_grad = False

        if use_lora:
            try:
                from peft import get_peft_model, LoraConfig, TaskType

                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                    lora_dropout=0.05,
                    bias='none',
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                self.vlm = get_peft_model(self.vlm, lora_config)
                logger.info(f'Applied LoRA: r={lora_rank}, alpha={lora_alpha}')
            except ImportError:
                logger.warning('peft not installed, skipping LoRA')

    def _prepare_inputs(self, images: List[Image.Image], goals: List[str]):
        messages_batch = []

        for image, goal in zip(images, goals):
            messages = [{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': f'Goal: {goal}\nWhat action should be taken?'},
                ],
            }]
            messages_batch.append(messages)

        texts = []
        for messages in messages_batch:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

        inputs = self.processor(
            text=texts,
            images=list(images),
            return_tensors='pt',
            padding=True,
        )

        return {key: value.to(self.device) for key, value in inputs.items()}

    def forward(
        self,
        images: List[Image.Image],
        goals: List[str],
    ) -> Dict[str, torch.Tensor]:
        inputs = self._prepare_inputs(images, goals)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.vlm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden = outputs.hidden_states[-1]
        last_hidden = hidden[:, -1, :].float()

        action_features = self.action_head(last_hidden)

        action_type_logits = self.action_type_head(action_features)
        action_param_mean = self.action_param_head(action_features)
        action_param_std = self.action_param_logstd.exp().expand_as(action_param_mean)

        value = self.value_head(last_hidden).squeeze(-1)

        return {
            'action_type_logits': action_type_logits,
            'action_type_probs': F.softmax(action_type_logits, dim=-1),
            'param_mean': action_param_mean,
            'param_std': action_param_std,
            'value': value,
        }

    def sample_action(
        self,
        images: List[Image.Image],
        goals: List[str],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        output = self.forward(images, goals)

        if deterministic:
            action_type = output['action_type_probs'].argmax(dim=-1)
            params = output['param_mean']
        else:
            action_type = torch.distributions.Categorical(
                probs=output['action_type_probs']
            ).sample()
            params = torch.distributions.Normal(
                output['param_mean'], output['param_std']
            ).sample()

        action = torch.cat([action_type.unsqueeze(-1).float(), params], dim=-1)

        type_log_prob = torch.distributions.Categorical(
            probs=output['action_type_probs']
        ).log_prob(action_type)

        param_log_prob = torch.distributions.Normal(
            output['param_mean'], output['param_std']
        ).log_prob(params).sum(dim=-1)

        log_prob = type_log_prob+param_log_prob

        return action, log_prob, output

    def compute_bc_loss(
        self,
        images: List[Image.Image],
        goals: List[str],
        target_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        output = self.forward(images, goals)

        target_type = target_actions[:, 0].long()
        type_loss = F.cross_entropy(output['action_type_logits'], target_type)

        target_params = target_actions[:, 1:]
        param_loss = F.mse_loss(output['param_mean'], target_params)

        total_loss = type_loss+0.1*param_loss

        return {
            'total': total_loss,
            'type_loss': type_loss,
            'param_loss': param_loss,
        }


class VLMDataset(Dataset):
    """Dataset for VLM training with rendered images."""

    def __init__(
        self,
        data_path: str,
        env: GoalConditionedEnv,
        max_samples: int = None,
    ):
        self.env = env
        self.data = []

        with open(data_path) as file:
            for index, line in enumerate(file):
                if max_samples and index >= max_samples:
                    break
                self.data.append(json.loads(line))

        logger.info(f'Loaded {len(self.data)} samples from {data_path}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        state_data = item['state']
        goal_type = item['goal_type']

        image = self._render_state(state_data)
        action = self._goal_to_action(goal_type, state_data)

        return {
            'image': image,
            'goal': item['natural_command'],
            'goal_type': goal_type,
            'action': action,
        }

    def _render_state(self, state_data: dict) -> Image.Image:
        self.env.reset()
        positions = state_data['positions']

        for index, position in enumerate(positions[:self.env.max_particles]):
            self.env.operator.sim.add_particle(position[0], position[1])

        frame = self.env.render()

        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame.astype(np.uint8))
        else:
            image = frame

        return image.resize((224, 224))

    def _goal_to_action(self, goal_type: str, state: dict) -> torch.Tensor:
        positions = np.array(state['positions'])
        center = positions.mean(axis=0) if len(positions) > 0 else np.array([32, 32])

        action = torch.zeros(7)

        if goal_type == 'directional_push':
            action[0] = ActionType.APPLY_FORCE.value
            action[2:4] = torch.tensor([32, 10])
            action[4] = 2.0
            action[5] = 40
        elif goal_type == 'cluster':
            action[0] = ActionType.APPLY_ATTRACTION.value
            action[2:4] = torch.tensor(center)
            action[4] = 3.0
            action[5] = 50
        elif goal_type == 'spread':
            action[0] = ActionType.APPLY_REPULSION.value
            action[2:4] = torch.tensor(center)
            action[4] = 3.0
            action[5] = 50
        elif goal_type in ['vibrate', 'heat']:
            action[0] = ActionType.APPLY_HEAT.value
            action[2:4] = torch.tensor(center)
            action[4] = 5.0
            action[5] = 50
        else:
            action[0] = ActionType.APPLY_FORCE.value
            action[2:4] = torch.tensor([32, 32])
            action[4] = 1.0
            action[5] = 20

        return action


def collate_vlm(batch):
    return {
        'images': [item['image'] for item in batch],
        'goals': [item['goal'] for item in batch],
        'actions': torch.stack([item['action'] for item in batch]),
    }


class RolloutBuffer:
    """On-policy rollout buffer for PPO."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.images = []
        self.goals = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.returns = []
        self.advantages = []

    def add(self, image, goal, action, log_prob, reward, value, done):
        self.images.append(image)
        self.goals.append(goal)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.images)

    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        returns, advantages = [], []
        gae = 0

        for step in reversed(range(len(self.rewards))):
            next_value = 0 if step == len(self.rewards)-1 else self.values[step+1]
            delta = self.rewards[step]+gamma*next_value*(1-self.dones[step])-self.values[step]
            gae = delta+gamma*gae_lambda*(1-self.dones[step])*gae
            advantages.insert(0, gae)
            returns.insert(0, gae+self.values[step])

        self.returns = returns
        self.advantages = advantages

    def get_batches(self, batch_size: int, device: str):
        indices = np.random.permutation(len(self.images))

        for start in range(0, len(self.images), batch_size):
            batch_indices = indices[start:start+batch_size]
            yield {
                'images': [self.images[index] for index in batch_indices],
                'goals': [self.goals[index] for index in batch_indices],
                'actions': torch.stack([self.actions[index] for index in batch_indices]).to(device),
                'log_probs': torch.stack([self.log_probs[index] for index in batch_indices]).to(device),
                'returns': torch.tensor([self.returns[index] for index in batch_indices], device=device),
                'advantages': torch.tensor([self.advantages[index] for index in batch_indices], device=device),
            }


def get_state_from_env(env) -> SimState:
    sim = env.operator.sim
    active_mask = sim.active
    particle_count = int(active_mask.sum())

    if particle_count == 0:
        return SimState(0, [], [])

    positions = sim.positions[active_mask][:particle_count].tolist()
    velocities = sim.velocities[active_mask][:particle_count].tolist()

    return SimState(particle_count, positions, velocities)


def compute_dense_reward(goal_dsl, state_before: SimState, state_after: SimState) -> float:
    if not state_after.positions or not state_before.positions:
        return -0.1

    positions_before = np.array(state_before.positions)
    positions_after = np.array(state_after.positions)
    velocities_after = (
        np.array(state_after.velocities)
        if state_after.velocities
        else np.zeros_like(positions_after)
    )

    goal_type = goal_dsl.__class__.__name__

    if goal_dsl.check_success(state_before, state_after):
        return 1.0

    if goal_type == 'DirectionalPushGoal':
        direction = goal_dsl.direction
        direction_vector = np.array(DIRECTION_VECTORS.get(direction, [0, 0]))
        movement = positions_after.mean(axis=0)-positions_before.mean(axis=0)
        progress = np.dot(movement, direction_vector)
        return np.clip(progress/5.0, -0.5, 0.5)

    elif goal_type == 'ClusterGoal':
        spread_before = positions_before.std() if len(positions_before) > 1 else 0
        spread_after = positions_after.std() if len(positions_after) > 1 else 0
        improvement = spread_before-spread_after
        return np.clip(improvement/10.0, -0.3, 0.3)

    elif goal_type == 'SpreadGoal':
        spread_before = positions_before.std() if len(positions_before) > 1 else 0
        spread_after = positions_after.std() if len(positions_after) > 1 else 0
        improvement = spread_after-spread_before
        return np.clip(improvement/10.0, -0.3, 0.3)

    elif goal_type == 'VibrateGoal':
        velocity_magnitude = (
            np.linalg.norm(velocities_after, axis=1).mean()
            if len(velocities_after) > 0
            else 0
        )
        return np.clip(velocity_magnitude/5.0-0.1, -0.3, 0.3)

    return -0.05


def render_env_to_image(env) -> Image.Image:
    frame = env.render()

    if isinstance(frame, np.ndarray):
        image = Image.fromarray(frame.astype(np.uint8))
    else:
        image = frame

    return image.resize((224, 224))


def collect_ppo_rollouts(
    model: VLMPolicy,
    env: GoalConditionedEnv,
    buffer: RolloutBuffer,
    num_steps: int,
    device: str,
) -> Dict[str, float]:
    model.eval()

    observation = env.reset()

    goal_types = list(GOAL_GENERATORS.keys())
    goal_type = np.random.choice(goal_types)
    goal_dsl = GOAL_GENERATORS[goal_type](SimState.random())
    commands = NATURAL_COMMANDS.get(goal_type, ['do something'])
    goal_text = np.random.choice(commands)

    state_before = get_state_from_env(env)

    total_reward = 0
    success_count = 0
    episode_count = 0
    step_in_episode = 0

    for step in range(num_steps):
        image = render_env_to_image(env)

        with torch.no_grad():
            action, log_prob, info = model.sample_action([image], [goal_text])
            value = info['value'].item()

        action_numpy = action[0].cpu().numpy()
        env_action = np.zeros(7, dtype=np.float32)
        env_action[0] = int(action_numpy[0]) % 8
        env_action[1] = -1
        env_action[2:6] = np.clip(action_numpy[2:6], [0, 0, 0, 1], [64, 64, 10, 50])

        try:
            next_observation, _, done, _ = env.step(env_action)
        except Exception:
            done = True
            next_observation = observation

        state_after = get_state_from_env(env)
        reward = compute_dense_reward(goal_dsl, state_before, state_after)
        success = reward >= 0.9

        step_in_episode += 1
        done = done or success or step_in_episode >= 50

        buffer.add(
            image,
            goal_text,
            action.squeeze(0).cpu(),
            log_prob.squeeze(0).cpu(),
            reward,
            value,
            done,
        )

        total_reward += reward
        if success:
            success_count += 1

        observation = next_observation
        state_before = state_after

        if done:
            observation = env.reset()
            goal_type = np.random.choice(goal_types)
            goal_dsl = GOAL_GENERATORS[goal_type](SimState.random())
            commands = NATURAL_COMMANDS.get(goal_type, ['do something'])
            goal_text = np.random.choice(commands)
            state_before = get_state_from_env(env)
            step_in_episode = 0
            episode_count += 1

    return {
        'mean_reward': total_reward/num_steps,
        'success_rate': success_count/max(episode_count, 1),
        'episode_count': episode_count,
    }


def train_ppo_epoch(
    model: VLMPolicy,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    config,
    device: str,
) -> Dict[str, float]:
    model.train()
    buffer.compute_gae(config.rl.gamma, config.rl.gae_lambda)

    losses = {'policy': 0, 'value': 0, 'total': 0}
    update_count = 0

    for _ in range(config.rl.ppo_epochs):
        for batch in buffer.get_batches(config.train.batch_size, device):
            advantages = batch['advantages']
            if advantages.std() > 1e-8:
                advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)

            output = model.forward(batch['images'], batch['goals'])

            action_type = batch['actions'][:, 0].long()
            params = batch['actions'][:, 1:]

            type_log_prob = torch.distributions.Categorical(
                probs=output['action_type_probs']
            ).log_prob(action_type)
            param_log_prob = torch.distributions.Normal(
                output['param_mean'], output['param_std']
            ).log_prob(params).sum(dim=-1)
            log_prob = type_log_prob+param_log_prob

            ratio = (log_prob-batch['log_probs']).exp()
            clipped_ratio = torch.clamp(ratio, 1-config.rl.clip_ratio, 1+config.rl.clip_ratio)
            policy_loss = -torch.min(ratio*advantages, clipped_ratio*advantages).mean()

            value_loss = F.mse_loss(output['value'], batch['returns'])

            total_loss = policy_loss+0.5*value_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses['policy'] += policy_loss.item()
            losses['value'] += value_loss.item()
            losses['total'] += total_loss.item()
            update_count += 1

    return {key: value/max(update_count, 1) for key, value in losses.items()}


def train_behavior_cloning(
    model: nn.Module,
    env: GoalConditionedEnv,
    config,
    dist_info: dict,
    checkpoint_dir: Path,
) -> None:
    if is_main_process(dist_info):
        logger.info('='*70)
        logger.info('Phase 1: Behavior Cloning with VLM')
        logger.info('='*70)

    data_path = '/workspace/Projects/FlexibleWorld/data/goal_commands_v2.jsonl'
    if not Path(data_path).exists():
        if is_main_process(dist_info):
            logger.warning(f'BC data not found: {data_path}')
        return

    dataset = VLMDataset(data_path, env, max_samples=config.vlm.bc_max_samples)

    sampler = DistributedSampler(dataset) if dist_info['world_size'] > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size//4,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_vlm,
        num_workers=2,
    )

    base_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    device = dist_info['device']

    for epoch in range(config.vlm.bc_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0

        progress_bar = tqdm(
            dataloader,
            desc=f'BC Epoch {epoch+1}/{config.vlm.bc_epochs}',
            disable=not is_main_process(dist_info),
        )
        for batch in progress_bar:
            losses = base_model.compute_bc_loss(
                batch['images'],
                batch['goals'],
                batch['actions'].to(device),
            )

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += losses['total'].item()
            if is_main_process(dist_info):
                progress_bar.set_postfix({'loss': f"{losses['total'].item():.4f}"})

        average_loss = total_loss/len(dataloader)
        if is_main_process(dist_info):
            logger.info(f'BC Epoch {epoch+1}/{config.vlm.bc_epochs}: loss={average_loss:.4f}')

    if is_main_process(dist_info):
        save_checkpoint(base_model, optimizer, config.vlm.bc_epochs, checkpoint_dir/'vlm_bc.pt', None)


def train_ppo(
    model: nn.Module,
    env: GoalConditionedEnv,
    config,
    dist_info: dict,
    checkpoint_dir: Path,
) -> None:
    if is_main_process(dist_info):
        logger.info('='*70)
        logger.info('Phase 2: PPO Fine-tune with VLM')
        logger.info('='*70)

    base_model = model.module if isinstance(model, DDP) else model
    device = dist_info['device']

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr=config.train.lr*0.1,
        weight_decay=config.train.weight_decay,
    )

    buffer = RolloutBuffer()
    best_success_rate = 0

    for epoch in range(config.vlm.ppo_epochs):
        buffer.reset()
        rollout_info = collect_ppo_rollouts(base_model, env, buffer, config.vlm.rollout_steps, device)

        rollout_info = reduce_dict(rollout_info, dist_info)

        losses = train_ppo_epoch(model, buffer, optimizer, config, device)

        if is_main_process(dist_info):
            logger.info(
                f"PPO Epoch {epoch+1}/{config.vlm.ppo_epochs}: "
                f"reward={rollout_info['mean_reward']:.3f}, "
                f"success={rollout_info['success_rate']:.1%}, "
                f"policy_loss={losses['policy']:.4f}"
            )

            if rollout_info['success_rate'] > best_success_rate:
                best_success_rate = rollout_info['success_rate']
                save_checkpoint(base_model, optimizer, epoch+1, checkpoint_dir/'vlm_best.pt', None)
                logger.info(f'  -> New best! {best_success_rate:.1%}')

    if is_main_process(dist_info):
        save_checkpoint(base_model, optimizer, config.vlm.ppo_epochs, checkpoint_dir/'vlm_final.pt', None)
        logger.info('='*70)
        logger.info('VLM Policy Training Complete!')
        logger.info(f'Best success rate: {best_success_rate:.1%}')
        logger.info('='*70)


@scope
def main(config):
    """VLM Policy Training Pipeline with DDP support."""
    dist_info = setup_distributed()
    device = dist_info['device']

    try:
        if is_main_process(dist_info):
            logger.info(f'Device: {device}')
            logger.info(f'GPUs available: {torch.cuda.device_count()}')
            logger.info(f'World size: {dist_info["world_size"]}')
            logger.info('='*70)
            logger.info('Creating VLM Policy (Qwen2-VL)')
            logger.info('='*70)

        vlm_name = getattr(config.model.vlm, 'name', 'Qwen/Qwen2-VL-2B-Instruct')

        model = VLMPolicy(
            model_name=vlm_name,
            action_dim=8,
            action_param_dim=6,
            use_lora=config.model.vlm.use_lora,
            lora_rank=config.model.vlm.lora_r,
            lora_alpha=config.model.vlm.lora_alpha,
            device=device,
        )

        if dist_info['world_size'] > 1:
            model = DDP(model, device_ids=[dist_info['local_rank']], find_unused_parameters=True)
            if is_main_process(dist_info):
                logger.info(f'Wrapped model with DDP (world_size={dist_info["world_size"]})')

        base_model = model.module if isinstance(model, DDP) else model
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

        if is_main_process(dist_info):
            logger.info(f'Total parameters: {total_params:,}')
            logger.info(f'Trainable parameters: {trainable_params:,}')

        checkpoint_dir = Path(config.storage.checkpoint_dir)
        if is_main_process(dist_info):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if dist_info['world_size'] > 1:
            dist.barrier()

        env = GoalConditionedEnv(
            width=config.sim.width,
            height=config.sim.height,
            max_particles=config.env.max_particles,
        )

        if config.vlm.do_bc:
            train_behavior_cloning(model, env, config, dist_info, checkpoint_dir)

        train_ppo(model, env, config, dist_info, checkpoint_dir)

    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
