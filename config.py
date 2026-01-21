"""
FlexibleWorld - ato scope configuration
Goal-Conditioned RL World Model
"""
from ato.scope import Scope
from ato.adict import ADict

scope = Scope(ADict.auto())


# =============================================================================
# Default Configuration
# =============================================================================

@scope.observe(default=True)
def default(config: ADict):
    # -------------------------------------------------------------------------
    # Simulator
    # -------------------------------------------------------------------------
    config.sim.width = 64
    config.sim.height = 64
    config.sim.max_particles = 256

    # physics parameters
    config.sim.physics.gravity_y = 0.0
    config.sim.physics.restitution = 0.8
    config.sim.physics.em_force_strength = 5.0
    config.sim.physics.em_force_radius = 15.0
    config.sim.physics.substeps = 3
    config.sim.physics.vibration_scale = 0.5
    config.sim.physics.heat_dissipation = 0.02
    config.sim.physics.thermal_radiation_rate = 0.01

    # action space
    config.sim.action.n_action_types = 8  # NOOP, ADD_PARTICLE, SET_PROPERTY, etc.
    config.sim.action.action_dim = 7  # action vector dimension
    config.sim.action.move_speed = 2.0
    config.sim.action.push_force = 3.0
    config.sim.action.pull_force = 2.0
    config.sim.action.interaction_radius = 8.0

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------
    config.env.max_steps = 200
    config.env.max_particles = 50
    config.env.obs_type = 'image'  # 'image' or 'state'
    config.env.render_scale = 4
    config.env.n_initial_particles = 8

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    config.data.path = 'data/'
    config.data.num_workers = 4

    config.data.collect.n_episodes = 100
    config.data.collect.n_steps_per_episode = 100
    config.data.collect.n_particles_range = [5, 20]
    config.data.collect.render_scale = 1
    config.data.collect.output_dir = 'data/simulator'

    # -------------------------------------------------------------------------
    # Model Architecture (GoalConditionedWorldModel)
    # -------------------------------------------------------------------------
    config.model.embed_dim = 768
    config.model.hidden_dim = 512
    config.model.image_size = 64
    config.model.channels = 3
    config.model.state_dim = None  # None = use image, int = use vector state

    # vision encoder (state encoder)
    config.model.vision.type = 'cnn'  # cnn, clip, dinov2
    config.model.vision.name = 'openai/clip-vit-base-patch32'
    config.model.vision.freeze = True
    config.model.vision.dim = 768

    # goal encoder (LLM or simple embedding)
    config.model.goal.type = 'embed'  # embed, llm
    config.model.goal.llm_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    config.model.goal.freeze = False  # train with LoRA
    config.model.goal.load_in_8bit = True
    config.model.goal.dim = 768
    config.model.goal.n_types = 16  # number of goal type embeddings

    # policy type
    config.model.policy_type = 'mlp'  # mlp, vlm

    # VLM policy (when policy_type='vlm')
    config.model.vlm.name = 'Qwen/Qwen2-VL-2B-Instruct'
    config.model.vlm.dim = 1536
    config.model.vlm.load_in_4bit = True
    config.model.vlm.use_lora = True
    config.model.vlm.lora_r = 16
    config.model.vlm.lora_alpha = 32
    config.model.vlm.lora_dropout = 0.05
    config.model.vlm.lora_target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    config.train.batch_size = 32
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-6
    config.train.grad_clip = 100.0
    config.train.epochs = 100
    config.train.save_every = 10
    config.train.log_every = 100

    # -------------------------------------------------------------------------
    # RL Training
    # -------------------------------------------------------------------------
    config.rl.mode = 'bc'  # 'bc' (behavior cloning), 'ppo', 'dpo'
    config.rl.gamma = 0.99
    config.rl.gae_lambda = 0.95
    config.rl.clip_ratio = 0.2
    config.rl.ppo_epochs = 4
    config.rl.rollout_steps = 2048
    config.rl.entropy_coef = 0.01
    config.rl.value_coef = 0.5

    # model components
    config.rl.use_value_head = True
    config.rl.use_dynamics = True
    config.rl.use_reward_predictor = True
    config.rl.use_frame_decoder = False

    # loss weights
    config.rl.loss.policy_weight = 1.0
    config.rl.loss.value_weight = 0.5
    config.rl.loss.dynamics_weight = 0.1
    config.rl.loss.reward_weight = 0.5
    config.rl.loss.entropy_weight = 0.01

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    config.inference.model_path = None
    config.inference.output_path = 'rollout.gif'
    config.inference.num_steps = 50
    config.inference.fps = 10
    config.inference.deterministic = False

    # -------------------------------------------------------------------------
    # Storage Paths (all on /workspace for container)
    # -------------------------------------------------------------------------
    config.storage.base_dir = '/workspace/Projects/FlexibleWorld'
    config.storage.cache_dir = '/workspace/.cache'
    config.storage.hf_home = '/workspace/.cache/huggingface'
    config.storage.data_dir = '/workspace/Projects/FlexibleWorld/data'
    config.storage.checkpoint_dir = '/workspace/Projects/FlexibleWorld/checkpoints'
    config.storage.output_dir = '/workspace/Projects/FlexibleWorld/outputs'

    # -------------------------------------------------------------------------
    # Experiment Tracking
    # -------------------------------------------------------------------------
    config.experiment.project_name = 'flexible_world'
    config.experiment.run_name = None
    config.experiment.checkpoint_dir = '/workspace/Projects/FlexibleWorld/checkpoints'
    config.experiment.use_wandb = False

    # -------------------------------------------------------------------------
    # VLM Policy Training
    # -------------------------------------------------------------------------
    config.vlm.do_bc = False
    config.vlm.bc_epochs = 5
    config.vlm.bc_max_samples = 1000
    config.vlm.ppo_epochs = 50
    config.vlm.rollout_steps = 256


# =============================================================================
# Vision Encoder Presets
# =============================================================================

@scope.observe(priority=1)
def clip_encoder(config: ADict):
    """Use CLIP as vision encoder"""
    config.model.vision.type = 'clip'
    config.model.vision.name = 'openai/clip-vit-base-patch32'
    config.model.vision.dim = 512


@scope.observe(priority=1)
def dinov2_encoder(config: ADict):
    """Use DINOv2 as vision encoder"""
    config.model.vision.type = 'dinov2'
    config.model.vision.name = 'dinov2_vitb14'
    config.model.vision.dim = 768


@scope.observe(priority=1)
def cnn_encoder(config: ADict):
    """Use simple CNN encoder (default, no pretrained)"""
    config.model.vision.type = 'cnn'
    config.model.vision.freeze = False


# =============================================================================
# Goal Encoder Presets
# =============================================================================

@scope.observe(priority=1)
def llm_goal_encoder(config: ADict):
    """Use LLM for goal encoding"""
    config.model.goal.type = 'llm'


@scope.observe(priority=2, chain_with='llm_goal_encoder')
def llama_goal(config: ADict):
    """Use LLaMA for goal encoding"""
    config.model.goal.llm_name = 'meta-llama/Llama-2-7b-hf'
    config.model.goal.dim = 4096


@scope.observe(priority=2, chain_with='llm_goal_encoder')
def mistral_goal(config: ADict):
    """Use Mistral for goal encoding"""
    config.model.goal.llm_name = 'mistralai/Mistral-7B-v0.1'
    config.model.goal.dim = 4096


@scope.observe(priority=2, chain_with='llm_goal_encoder')
def qwen_goal(config: ADict):
    """Qwen2.5-1.5B for goal encoding (recommended for Korean)"""
    config.model.goal.llm_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    config.model.goal.dim = 1536
    config.model.goal.load_in_8bit = True


@scope.observe(priority=2, chain_with='llm_goal_encoder')
def qwen3b_goal(config: ADict):
    """Qwen2.5-3B for stronger goal understanding"""
    config.model.goal.llm_name = 'Qwen/Qwen2.5-3B-Instruct'
    config.model.goal.dim = 2048
    config.model.goal.load_in_8bit = True


@scope.observe(priority=2, chain_with='llm_goal_encoder')
def phi3_goal(config: ADict):
    """Phi-3-mini for goal encoding"""
    config.model.goal.llm_name = 'microsoft/Phi-3-mini-4k-instruct'
    config.model.goal.dim = 3072


@scope.observe(priority=2, chain_with='llm_goal_encoder')
def gemma_goal(config: ADict):
    """Gemma-2-2B for goal encoding"""
    config.model.goal.llm_name = 'google/gemma-2-2b-it'
    config.model.goal.dim = 2304
    config.model.goal.load_in_8bit = True


@scope.observe(priority=2, chain_with='llm_goal_encoder')
def llama3_goal(config: ADict):
    """Llama-3.2-1B for lightweight goal encoding"""
    config.model.goal.llm_name = 'meta-llama/Llama-3.2-1B-Instruct'
    config.model.goal.dim = 2048


# =============================================================================
# VLM Policy Presets (Image + Goal → Action directly)
# =============================================================================

@scope.observe(priority=1)
def vlm_policy(config: ADict):
    """Use VLM as full policy (image+goal → action)"""
    config.model.policy_type = 'vlm'
    config.model.vlm.load_in_4bit = True
    config.model.vlm.use_lora = True
    config.model.vlm.lora_r = 16
    config.model.vlm.lora_alpha = 32


@scope.observe(priority=2, chain_with='vlm_policy')
def qwen_vlm(config: ADict):
    """Qwen2-VL-2B (recommended: vision + Korean)"""
    config.model.vlm.name = 'Qwen/Qwen2-VL-2B-Instruct'
    config.model.vlm.dim = 1536


@scope.observe(priority=2, chain_with='vlm_policy')
def qwen_vlm_7b(config: ADict):
    """Qwen2-VL-7B for stronger vision understanding"""
    config.model.vlm.name = 'Qwen/Qwen2-VL-7B-Instruct'
    config.model.vlm.dim = 3584


@scope.observe(priority=2, chain_with='vlm_policy')
def internvl_vlm(config: ADict):
    """InternVL2-2B for vision-language tasks"""
    config.model.vlm.name = 'OpenGVLab/InternVL2-2B'
    config.model.vlm.dim = 2048


@scope.observe(priority=2, chain_with='vlm_policy')
def phi3_vlm(config: ADict):
    """Phi-3.5-vision for efficient vision-language"""
    config.model.vlm.name = 'microsoft/Phi-3.5-vision-instruct'
    config.model.vlm.dim = 3072


# =============================================================================
# Simulator/Environment Presets
# =============================================================================

@scope.observe(priority=1)
def high_res(config: ADict):
    """High resolution (128x128)"""
    config.sim.width = 128
    config.sim.height = 128
    config.model.image_size = 128
    config.train.batch_size = 16


@scope.observe(priority=1)
def small_sim(config: ADict):
    """Small simulation for faster iteration"""
    config.sim.width = 32
    config.sim.height = 32
    config.model.image_size = 32
    config.sim.max_particles = 64
    config.env.max_particles = 20


@scope.observe(priority=1)
def zero_gravity(config: ADict):
    """Zero gravity physics"""
    config.sim.physics.gravity_y = 0.0


@scope.observe(priority=1)
def gravity(config: ADict):
    """Standard downward gravity"""
    config.sim.physics.gravity_y = 0.1


@scope.observe(priority=1)
def elastic(config: ADict):
    """Nearly elastic collisions"""
    config.sim.physics.restitution = 0.95


@scope.observe(priority=1)
def strong_em(config: ADict):
    """Strong electromagnetic forces"""
    config.sim.physics.em_force_strength = 10.0
    config.sim.physics.em_force_radius = 25.0


# =============================================================================
# RL Training Presets
# =============================================================================

@scope.observe(priority=1)
def rl_bc(config: ADict):
    """Behavior cloning from demonstrations"""
    config.rl.mode = 'bc'


@scope.observe(priority=1)
def rl_ppo(config: ADict):
    """Online RL with PPO"""
    config.rl.mode = 'ppo'
    config.rl.rollout_steps = 2048
    config.rl.ppo_epochs = 4
    # Use vector state for particle simulation (8 particles × 2 pos + 2 vel = 32)
    config.model.state_dim = 32
    config.model.embed_dim = 256
    config.model.hidden_dim = 128


@scope.observe(priority=1)
def rl_dpo(config: ADict):
    """Direct Preference Optimization (offline)"""
    config.rl.mode = 'dpo'
    config.rl.use_dynamics = False


@scope.observe(priority=2, chain_with='rl_ppo')
def rl_fast(config: ADict):
    """Fast RL iteration for debugging"""
    config.rl.rollout_steps = 256
    config.rl.ppo_epochs = 2
    config.train.epochs = 10


@scope.observe(priority=2, chain_with='rl_ppo')
def rl_long(config: ADict):
    """Long RL training"""
    config.rl.rollout_steps = 4096
    config.train.epochs = 1000


# =============================================================================
# Training Presets
# =============================================================================

@scope.observe(priority=1)
def debug(config: ADict):
    """Debug mode with small model and fast iteration"""
    config.model.embed_dim = 256
    config.model.hidden_dim = 128
    config.train.batch_size = 4
    config.train.log_every = 10
    config.train.save_every = 1
    config.train.epochs = 5
    config.rl.rollout_steps = 128


@scope.observe(priority=1)
def fast_train(config: ADict):
    """Fast training with larger batch"""
    config.train.batch_size = 64
    config.train.lr = 3e-4


@scope.observe(priority=1)
def long_train(config: ADict):
    """Long training run"""
    config.train.epochs = 500
    config.train.save_every = 50


# =============================================================================
# Data Collection Presets
# =============================================================================

@scope.observe(priority=1)
def collect_small(config: ADict):
    """Small data collection for testing"""
    config.data.collect.n_episodes = 10
    config.data.collect.n_steps_per_episode = 50


@scope.observe(priority=1)
def collect_large(config: ADict):
    """Large data collection for training"""
    config.data.collect.n_episodes = 1000
    config.data.collect.n_steps_per_episode = 200


# =============================================================================
# vLLM Data Generation Presets
# =============================================================================

@scope.observe(priority=1)
def gen_data(config: ADict):
    """Enable vLLM data generation mode"""
    config.gen.enabled = True
    config.gen.vllm_url = 'http://localhost:8000/v1'
    config.gen.model = 'Qwen/Qwen2.5-72B-Instruct'
    config.gen.n_episodes = 10000
    config.gen.n_workers = 16
    config.gen.use_image = False
    config.gen.output_path = '/workspace/Projects/FlexibleWorld/data/train_generated.jsonl'


@scope.observe(priority=2, chain_with='gen_data')
def gen_small(config: ADict):
    """Small generation for testing"""
    config.gen.n_episodes = 100
    config.gen.n_workers = 4


@scope.observe(priority=2, chain_with='gen_data')
def gen_large(config: ADict):
    """Large generation (100K samples)"""
    config.gen.n_episodes = 100000
    config.gen.n_workers = 32


@scope.observe(priority=2, chain_with='gen_data')
def gen_with_image(config: ADict):
    """Generate with image in prompt (slower but more accurate)"""
    config.gen.use_image = True
    config.gen.n_workers = 8  # slower, use fewer workers


@scope.observe(priority=2, chain_with='gen_data')
def gen_qwen32b(config: ADict):
    """Use Qwen2.5-32B (faster)"""
    config.gen.model = 'Qwen/Qwen2.5-32B-Instruct'


@scope.observe(priority=2, chain_with='gen_data')
def gen_llama70b(config: ADict):
    """Use Llama-3.1-70B"""
    config.gen.model = 'meta-llama/Llama-3.1-70B-Instruct'


# =============================================================================
# Combined Presets
# =============================================================================

@scope.observe(priority=1)
def full_model(config: ADict):
    """Full model with all components enabled"""
    config.rl.use_value_head = True
    config.rl.use_dynamics = True
    config.rl.use_reward_predictor = True
    config.rl.use_frame_decoder = True


@scope.observe(priority=1)
def policy_only(config: ADict):
    """Policy-only model (no dynamics/reward prediction)"""
    config.rl.use_value_head = True
    config.rl.use_dynamics = False
    config.rl.use_reward_predictor = False
    config.rl.use_frame_decoder = False


# =============================================================================
# VLM Policy Presets
# =============================================================================

@scope.observe(priority=1)
def vlm_full(config: ADict):
    """Full VLM policy training: BC + PPO"""
    config.vlm.do_bc = True
    config.vlm.bc_epochs = 5
    config.vlm.bc_max_samples = 1000
    config.vlm.ppo_epochs = 50
    config.vlm.rollout_steps = 256
    config.model.vlm.name = 'Qwen/Qwen2-VL-2B-Instruct'
    config.model.vlm.use_lora = True
    config.model.vlm.lora_r = 16
    config.model.vlm.lora_alpha = 32
    config.train.batch_size = 8  # Smaller for VLM


@scope.observe(priority=1)
def vlm_fast(config: ADict):
    """Fast VLM training (testing)"""
    config.vlm.do_bc = False
    config.vlm.ppo_epochs = 10
    config.vlm.rollout_steps = 64
    config.model.vlm.name = 'Qwen/Qwen2-VL-2B-Instruct'
    config.model.vlm.use_lora = True
    config.train.batch_size = 4


@scope.observe(priority=1)
def vlm_7b(config: ADict):
    """Use larger Qwen2-VL-7B model"""
    config.model.vlm.name = 'Qwen/Qwen2-VL-7B-Instruct'
    config.model.vlm.lora_r = 32
    config.train.batch_size = 4


@scope.observe(priority=1)
def vlm_long(config: ADict):
    """Long VLM training"""
    config.vlm.do_bc = True
    config.vlm.bc_epochs = 10
    config.vlm.bc_max_samples = 5000
    config.vlm.ppo_epochs = 100
    config.vlm.rollout_steps = 512


# =============================================================================
# Manual Entries
# =============================================================================

@scope.manual
def sim_width(config: ADict):
    return 'Simulation width in pixels'


@scope.manual
def sim_height(config: ADict):
    return 'Simulation height in pixels'


@scope.manual
def model_vision_type(config: ADict):
    return 'Vision encoder type: cnn, clip, dinov2'


@scope.manual
def model_goal_type(config: ADict):
    return 'Goal encoder type: embed (simple), llm (language model)'


@scope.manual
def rl_mode(config: ADict):
    return 'RL training mode: bc (behavior cloning), ppo, dpo'


@scope.manual
def train_lr(config: ADict):
    return 'Learning rate for training'


@scope.manual
def inference_model_path(config: ADict):
    return 'Path to checkpoint for inference'


@scope.manual
def env_obs_type(config: ADict):
    return 'Observation type: image (RGB) or state (vector)'


@scope.manual
def model_policy_type(config: ADict):
    return 'Policy type: mlp (state+goal embeddings), vlm (end-to-end VLM)'


@scope.manual
def model_vlm_name(config: ADict):
    return 'VLM model name for vlm policy type'


@scope.manual
def model_vlm_lora_r(config: ADict):
    return 'LoRA rank for VLM fine-tuning'
