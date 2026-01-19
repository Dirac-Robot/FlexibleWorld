# World Model Base

Latent 기반 월드 모델 학습/추론 프레임워크. DreamerV3 스타일의 RSSM (Recurrent State-Space Model) 아키텍처 기반.

## Features

- **Modular Architecture**: Encoder, Dynamics, Decoder 모듈 교체 가능
- **Pretrained Finetuning**: Local 체크포인트 또는 HuggingFace Hub에서 로드
- **Game-like Inference**: Action → Frame 렌더링 파이프라인
- **ato Config**: 유연한 configuration 시스템

## Installation

```bash
conda env create -f environment.yaml
conda activate world
```

## Quick Start

### Training

```bash
# Default training
python train.py data.path=path/to/data

# Debug mode (small model)
python train.py debug data.path=path/to/data

# High resolution
python train.py high_res data.path=path/to/data
```

### Finetuning

```bash
# From local checkpoint
python finetune.py finetune finetune.pretrained_path=checkpoints/epoch_100.pt data.path=path/to/data

# From HuggingFace Hub
python finetune.py finetune finetune_from_hub finetune.hub_repo=username/world-model data.path=path/to/data

# Freeze encoder only
python finetune.py finetune finetune.pretrained_path=model.pt finetune.freeze_modules="['encoder']"
```

### Inference

```bash
# Interactive rollout from checkpoint
python inference.py checkpoints/final.pt --output rollout.gif --steps 100

# From HuggingFace Hub
python inference.py username/world-model --hub --output rollout.gif
```

### Interactive Demo

```bash
python demo.py demo_mode demo.model_path=checkpoints/final.pt
```

## Architecture

### FlexibleWorldModel (Pretrained 플러그인)

```mermaid
graph TB
    subgraph Input
        OBS["Image (B,T,C,H,W)"]
        ACT["Actions (B,T)"]
    end

    subgraph VisionEncoder["Vision Encoder (CLIP/DINOv2/Custom)"]
        VE_IN[Input] --> VE_L1[Layer 1]
        VE_L1 --> VE_L2[Layer 2]
        VE_L2 --> VE_LN[Layer N]
        VE_LN --> VE_OUT[Output]
        VE_L1 -.-> VE_HOOK1[Hook]
        VE_L2 -.-> VE_HOOK2[Hook]
        VE_LN -.-> VE_HOOKN[Hook]
    end

    subgraph LLMBackbone["LLM Backbone (LLaMA/Mistral/Custom)"]
        LLM_IN[Input] --> LLM_L1[Layer 1]
        LLM_L1 --> LLM_L2[Layer 2]
        LLM_L2 --> LLM_LN[Layer N]
        LLM_LN --> LLM_OUT[Output]
        LLM_L1 -.-> LLM_HOOK1[Hook]
        LLM_L2 -.-> LLM_HOOK2[Hook]
        LLM_LN -.-> LLM_HOOKN[Hook]
    end

    subgraph VideoDecoder["Video Decoder (VAE/Custom)"]
        DEC_IN[Input] --> DEC_OUT[Output]
    end

    subgraph Manipulator["Layer Manipulator (Placeholder)"]
        PROC[Custom Processing]
        LOSS[Auxiliary Losses]
    end

    OBS --> VisionEncoder
    ACT --> ActionEmbed[Action Embedding]
    VE_OUT --> Proj[Projection]
    Proj --> LLM_IN
    ActionEmbed --> LLM_IN
    LLM_OUT --> DEC_IN
    DEC_OUT --> PRED["Predicted Frame (B,T,C,H,W)"]

    VE_HOOK1 -.-> PROC
    VE_HOOK2 -.-> PROC
    LLM_HOOK1 -.-> PROC
    LLM_HOOK2 -.-> PROC
    PROC -.-> LOSS
```

### WorldModel (RSSM - 기본)

```mermaid
graph LR
    IMG[Image] --> ENC[Encoder CNN]
    ENC --> EMB[Embedding]
    EMB --> RSSM[RSSM Dynamics]
    ACT[Action] --> RSSM
    RSSM --> DET[Deterministic h]
    RSSM --> STO[Stochastic z]
    DET --> DEC[Decoder CNN]
    STO --> DEC
    DEC --> PRED[Predicted Frame]
```

### Directory Structure

```
wm_base/
├── environment.yaml          # conda 'world' 환경
├── config.py                 # ato scope configuration
├── train.py                  # 학습 엔트리포인트
├── finetune.py               # pretrained finetuning
├── inference.py              # 추론 파이프라인
├── demo.py                   # 인터랙티브 데모
├── dataset.py                # 데이터 로딩
├── models/
│   ├── encoder.py            # CNN 인코더
│   ├── decoder.py            # Transposed CNN 디코더
│   ├── dynamics.py           # RSSM dynamics
│   ├── world_model.py        # WorldModel (RSSM)
│   ├── backbone.py           # Pretrained wrapper classes
│   │   ├── VisionEncoderWrapper
│   │   ├── LLMBackboneWrapper
│   │   ├── VideoDecoderWrapper
│   │   ├── LayerOutputCollector
│   │   └── LayerManipulator
│   └── flexible_world_model.py  # FlexibleWorldModel
└── utils/
    ├── checkpoint.py         # 체크포인트 유틸
    ├── visualization.py      # 시각화 (gif/mp4)
    └── hub.py                # HuggingFace Hub 통합
```

## Module Replacement

```python
from models import WorldModel

model = WorldModel(...)

# Get/set modules
encoder = model.get_module('encoder')
model.set_module('encoder', CustomEncoder(...))

# Freeze/unfreeze
model.freeze_module('encoder')
model.unfreeze_module('decoder')
```

## HuggingFace Hub

```python
from utils.hub import push_to_hub, create_from_pretrained

# Upload
push_to_hub(model, 'username/my-world-model')

# Download
model = create_from_pretrained('username/my-world-model')
```

## Data Format

HDF5 file with:
- `frames`: (N, T, C, H, W) or (N, T, H, W, C) float [0-255]
- `actions`: (N, T) int

Or directory of .npz files with `frames` and `actions` keys.

## Configuration

See `config.py` for all options. Available views:
- `default`: Standard settings
- `debug`: Small model for testing
- `finetune`: Lower LR, freeze encoder
- `high_res`: 128x128 images
- `atari`: Atari game preset

## License

MIT
