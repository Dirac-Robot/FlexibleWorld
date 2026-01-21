# FlexibleWorld

**VLM-based Goal-Conditioned World Agent for Particle Simulation**

Vision-Language Model (Qwen2-VL)을 사용하여 이미지와 자연어 goal을 직접 처리하는 범용 World Agent.

## 🎯 Overview

```
"입자들을 왼쪽으로 이동시켜" (Natural Language Goal)
              +
        [64x64 RGB Image]
              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     VLM Policy (Qwen2-VL)                       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Vision-Language Model                        │  │
│  │                                                           │  │
│  │   [Image Tokens]  +  [Goal Text Tokens]                   │  │
│  │              ↓                                            │  │
│  │      Transformer Layers (with LoRA)                       │  │
│  │              ↓                                            │  │
│  │         Hidden States                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ↓                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │  Action Type   │  │ Action Params  │  │     Value      │    │
│  │   (8 types)    │  │  (x,y,val,r)   │  │   Estimate     │    │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘    │
└──────────│───────────────────│───────────────────│──────────────┘
           ↓                   ↓                   ↓
      APPLY_FORCE         x=54, y=32           V(s,g)
                          value=2.0
```

## ✨ Features

- **Vision-Language Model**: Qwen2-VL로 이미지+텍스트 직접 처리
- **Natural Language Goals**: 한국어/영어 자연어 명령 지원
- **LoRA Fine-tuning**: 효율적인 학습 (5.7M trainable / 2.2B total)
- **Dense Rewards**: Goal 진행도 기반 세밀한 보상
- **Generalizable**: 다양한 환경에 적용 가능한 구조
- **PPO Training**: 실제 시뮬레이터와 상호작용하며 학습

## 🏗️ Architecture

### VLM Policy Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VLM Training Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Phase 1: Data Generation (vLLM)                                           │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  Goal DSL (structured)  →  vLLM (Qwen-72B)  →  Natural Language   │    │
│   │  "move_to(id=1, x>90)"      augmentation       "입자를 오른쪽으로"   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                     ↓                                       │
│   Phase 2: BC Pretrain (Optional)                                           │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  [Image] + [Goal Text]  →  Qwen2-VL (LoRA)  →  Action             │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                     ↓                                       │
│   Phase 3: PPO Fine-tune (Real Simulator)                                   │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  ┌─────────┐      ┌──────────────┐      ┌─────────────────────┐  │    │
│   │  │ Render  │ ───→ │  VLM Policy  │ ───→ │ Execute in Sim      │  │    │
│   │  │ Image   │      │  (action)    │      │ Dense Reward        │  │    │
│   │  └─────────┘      └──────────────┘      └─────────────────────┘  │    │
│   │       ↑                                            │              │    │
│   │       └────────────── PPO Update ←─────────────────┘              │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   Model: Qwen2-VL-2B-Instruct (LoRA: 5.7M trainable / 2.2B total)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Action Space

```
ActionType (8 types):
├── 0: NOOP           - 아무 동작 없음
├── 1: ADD_PARTICLE   - 입자 추가
├── 2: SET_PROPERTY   - 입자 속성 변경
├── 3: APPLY_HEAT     - 열 적용
├── 4: APPLY_FORCE    - 힘 적용 (밀기)
├── 5: APPLY_ATTRACTION - 인력 적용 (당기기)
├── 6: APPLY_REPULSION  - 척력 적용
└── 7: STEP           - 시뮬레이션 진행

Action Vector (7-dim):
[action_type, target, x, y, value, radius, property_type]
```

### Dense Reward System

```python
# Goal에 가까워지면 positive, 멀어지면 negative
DirectionalPushGoal: dot(movement, direction) → [-0.5, 0.5]
ClusterGoal:         spread_before - spread_after → [-0.3, 0.3]
SpreadGoal:          spread_after - spread_before → [-0.3, 0.3]
VibrateGoal:         velocity_magnitude → [-0.3, 0.3]
Success:             1.0
```

## 📦 Installation

```bash
# Environment setup
conda env create -f environment.yaml
conda activate world

# Required packages
pip install transformers peft accelerate
pip install qwen-vl-utils  # For Qwen2-VL

# For data generation
pip install vllm openai
```

## 🚀 Quick Start

### VLM Policy Training

```bash
# Fast test (10 PPO epochs)
python train_vlm.py vlm_fast

# Full training (BC + PPO)
python train_vlm.py vlm_full

# Long training (100 epochs)
python train_vlm.py vlm_long

# Use larger 7B model
python train_vlm.py vlm_full vlm_7b
```

### Data Generation (vLLM)

```bash
# 1. Start vLLM server
./scripts/start_vllm.sh qwen32b

# 2. Generate training data
python scripts/generate_training_data.py \
    --n-goals 2000 \
    --n-variations 5 \
    --output data/goal_commands.jsonl
```

### Inference

```bash
# Run trained VLM agent
python inference.py \
    --model checkpoints/vlm_best.pt \
    --goal "입자들을 모아줘"
```

## 🔧 Configuration

### VLM Training Presets

| Preset | Description |
|--------|-------------|
| `vlm_fast` | Quick test (10 PPO epochs, no BC) |
| `vlm_full` | Full training (BC 5ep + PPO 50ep) |
| `vlm_long` | Production (BC 10ep + PPO 100ep) |
| `vlm_7b` | Use Qwen2-VL-7B (larger model) |

### Key Parameters

```python
# Model
config.model.vlm.name = 'Qwen/Qwen2-VL-2B-Instruct'
config.model.vlm.use_lora = True
config.model.vlm.lora_r = 16
config.model.vlm.lora_alpha = 32

# Training
config.train.batch_size = 8  # Smaller for VLM
config.train.lr = 1e-4

# PPO
config.vlm.ppo_epochs = 50
config.vlm.rollout_steps = 256
config.rl.clip_ratio = 0.2
config.rl.gamma = 0.99
```

### Example Commands

```bash
# Custom training
python train_vlm.py vlm_full \
    train.lr=3e-4 \
    vlm.ppo_epochs=100 \
    model.vlm.lora_r=32

# Debug mode
python train_vlm.py vlm_fast debug

# Override VLM model
python train_vlm.py vlm_full model.vlm.name:=Qwen/Qwen2-VL-7B-Instruct
```

## 📁 Directory Structure

```
FlexibleWorld/
├── config.py                 # ato scope configuration
├── train_vlm.py              # VLM Policy training ⭐
├── inference.py              # Inference pipeline
├── dataset.py                # Data loading
│
├── models/
│   ├── goal_world_model.py   # GoalConditionedWorldModel
│   ├── backbone.py           # CLIP/DINOv2/LLM wrappers
│   └── ...
│
├── simulator/
│   ├── core.py               # ParticleSimulator
│   ├── action_operator.py    # ActionOperator
│   ├── goal_env.py           # GoalConditionedEnv
│   └── ...
│
├── scripts/
│   ├── generate_training_data.py  # vLLM data generation
│   └── start_vllm.sh              # vLLM server script
│
├── data/
│   └── goal_commands_v2.jsonl     # Generated training data
│
├── checkpoints/
│   ├── vlm_best.pt                # Best VLM checkpoint
│   └── vlm_final.pt               # Final VLM checkpoint
│
└── logs/
    └── vlm_*.log                  # Training logs
```

## 🎮 Supported Goals

Natural language goals in Korean/English:

| Goal Type | Examples |
|-----------|----------|
| Directional | "move left", "입자들을 오른쪽으로", "push up" |
| Clustering | "cluster together", "가운데로 모아", "group particles" |
| Scattering | "scatter", "흩어지게 해", "spread apart" |
| Temperature | "heat up", "가열해", "make it vibrate" |
| Position | "move to corner", "중앙으로" |

## 🔬 Usage Example

```python
from train_vlm import VLMPolicy
from simulator.goal_env import GoalConditionedEnv
from PIL import Image

# Create environment
env = GoalConditionedEnv(width=64, height=64)

# Load trained VLM
model = VLMPolicy.from_checkpoint('checkpoints/vlm_best.pt')

# Run episode
obs = env.reset()
goal = "입자들을 왼쪽으로 이동시켜"

for _ in range(100):
    # Render current state
    image = Image.fromarray(env.render())
    
    # Get action from VLM
    action, _, _ = model.sample_action([image], [goal])
    
    # Execute
    obs, reward, done, info = env.step(action[0].numpy())
    
    if done:
        break
```

## 📊 Performance

| Model | Success Rate | Trainable Params | Inference Speed |
|-------|-------------|------------------|-----------------|
| Qwen2-VL-2B | ~40% | 5.7M | ~15 FPS |
| Qwen2-VL-7B | TBD | ~10M | ~8 FPS |

## 🗺️ Roadmap

- [ ] Multi-GPU DDP training
- [ ] BC pretrain from expert demonstrations
- [ ] Transfer to other simulation environments
- [ ] Real robot deployment

## 📄 License

MIT
