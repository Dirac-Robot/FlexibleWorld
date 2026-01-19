"""
World Model Base - ato scope configuration
"""
from ato.scope import Scope
from ato.adict import ADict

scope = Scope()


@scope.observe(default=True)
def default(config: ADict):
    # Model architecture
    config.model.latent_dim = 256
    config.model.hidden_dim = 512
    config.model.stoch_dim = 32
    config.model.stoch_classes = 32  # categorical latent
    config.model.action_dim = 4
    config.model.image_size = 64
    config.model.channels = 3

    # Flexible model (new)
    config.model.use_flexible = False  # use FlexibleWorldModel
    config.model.encoder_type = 'placeholder'  # clip, dinov2, placeholder
    config.model.encoder_name = 'openai/clip-vit-base-patch32'
    config.model.encoder_layer_names = []
    config.model.freeze_encoder = True

    config.model.llm_type = 'placeholder'  # llama, mistral, hf, placeholder
    config.model.llm_name = 'meta-llama/Llama-2-7b-hf'
    config.model.llm_layer_names = []
    config.model.freeze_llm = True
    config.model.load_in_8bit = False

    config.model.decoder_type = 'placeholder'  # vae, placeholder
    config.model.decoder_name = 'stabilityai/sd-vae-ft-mse'
    config.model.decoder_layer_names = []
    config.model.freeze_decoder = True

    config.model.vision_dim = 768
    config.model.llm_dim = 768
    config.model.output_dim = 768

    # Encoder/Decoder (legacy RSSM model)
    config.model.encoder.channels = [64, 128, 256, 512]
    config.model.decoder.channels = [512, 256, 128, 64]

    # Training
    config.train.batch_size = 32
    config.train.seq_len = 16
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-6
    config.train.grad_clip = 100.0
    config.train.epochs = 100
    config.train.save_every = 10
    config.train.log_every = 100

    # Loss weights
    config.train.loss.recon_weight = 1.0
    config.train.loss.kl_weight = 0.1
    config.train.loss.dynamics_weight = 1.0

    # Data
    config.data.path = 'data/'
    config.data.num_workers = 4

    # Experiment tracking
    config.experiment.project_name = 'world_model'
    config.experiment.run_name = None
    config.experiment.checkpoint_dir = 'checkpoints/'
    config.experiment.use_wandb = False


@scope.observe(priority=1)
def debug(config: ADict):
    """Debug mode with small model and fast iteration"""
    config.model.latent_dim = 64
    config.model.hidden_dim = 128
    config.train.batch_size = 4
    config.train.seq_len = 8
    config.train.log_every = 10
    config.train.save_every = 1


@scope.observe(priority=1)
def finetune(config: ADict):
    """Finetuning mode with lower LR and frozen encoder"""
    config.train.lr = 1e-5
    config.finetune.enabled = True
    config.finetune.pretrained_path = None  # must be set via CLI
    config.finetune.freeze_modules = ['encoder']


@scope.observe(priority=1)
def high_res(config: ADict):
    """High resolution (128x128) mode"""
    config.model.image_size = 128
    config.model.encoder.channels = [32, 64, 128, 256, 512]
    config.model.decoder.channels = [512, 256, 128, 64, 32]
    config.train.batch_size = 16


@scope.observe(priority=1)
def atari(config: ADict):
    """Atari game preset"""
    config.model.action_dim = 18
    config.model.image_size = 84
    config.model.channels = 1  # grayscale


@scope.manual
def model_latent_dim(config: ADict):
    return 'Latent space dimension for world model states'


@scope.manual
def train_lr(config: ADict):
    return 'Learning rate for training'


@scope.manual
def finetune_pretrained_path(config: ADict):
    return 'Path to pretrained checkpoint for finetuning'


@scope.manual
def finetune_freeze_modules(config: ADict):
    return 'List of module names to freeze during finetuning: encoder, decoder, dynamics'
