"""
Finetuning script for World Model with pretrained weights
"""
import torch
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from config import scope
from models.world_model import create_world_model
from dataset import create_dataloader
from train import compute_loss
from utils import save_checkpoint, load_pretrained
from utils.hub import load_from_hub, create_from_pretrained


@scope
def finetune(config):
    """Finetuning with pretrained weights."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Load pretrained model
    if config.finetune.get('from_hub', False):
        if config.finetune.hub_repo is None:
            raise ValueError('Must set finetune.hub_repo via CLI when using finetune_from_hub')
        logger.info(f'Loading pretrained from HuggingFace Hub: {config.finetune.hub_repo}')
        model = create_from_pretrained(
            config.finetune.hub_repo,
            device=device,
            action_dim=config.model.action_dim,  # allow override for different action spaces
        )
    elif config.finetune.pretrained_path:
        logger.info(f'Loading pretrained from local: {config.finetune.pretrained_path}')
        model = create_world_model(config)
        load_pretrained(
            path=config.finetune.pretrained_path,
            model=model,
            freeze_modules=list(config.finetune.freeze_modules),
            device=device,
        )
    else:
        raise ValueError('Must set finetune.pretrained_path or use finetune_from_hub view')

    model = model.to(device)

    # Freeze specified modules
    for module_name in config.finetune.freeze_modules:
        model.freeze_module(module_name)
        frozen_params = sum(p.numel() for p in model.get_module(module_name).parameters())
        logger.info(f'Froze {module_name}: {frozen_params:,} parameters')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Trainable: {trainable_params:,} / {total_params:,} parameters')

    # Create dataloader
    dataloader = create_dataloader(
        path=config.data.path,
        batch_size=config.train.batch_size,
        seq_len=config.train.seq_len,
        image_size=config.model.image_size,
        num_workers=config.data.num_workers,
    )

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    # Checkpoint directory
    ckpt_dir = Path(config.experiment.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    for epoch in range(config.train.epochs):
        model.train()
        epoch_losses = {'total': 0, 'recon': 0, 'kl': 0}

        pbar = tqdm(dataloader, desc=f'Finetune Epoch {epoch+1}/{config.train.epochs}')
        for batch_idx, (frames, actions) in enumerate(pbar):
            frames = frames.to(device)
            actions = actions.to(device)

            # Forward pass
            outputs = model(frames, actions)

            # Compute loss
            losses = compute_loss(outputs, frames, config)

            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

            optimizer.step()

            # Logging
            for k, v in losses.items():
                epoch_losses[k] += v.item()

            if global_step%config.train.log_every == 0:
                pbar.set_postfix({
                    'loss': f'{losses["total"].item():.4f}',
                    'recon': f'{losses["recon"].item():.4f}',
                })

            global_step += 1

        # Epoch summary
        n_batches = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        logger.info(f'Epoch {epoch+1}: loss={epoch_losses["total"]:.4f}')

        # Save checkpoint
        if (epoch + 1)%config.train.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                path=ckpt_dir/f'finetune_epoch_{epoch+1:04d}.pt',
                config=dict(config),
            )

    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.train.epochs,
        path=ckpt_dir/'finetune_final.pt',
        config=dict(config),
    )
    logger.info('Finetuning complete!')


if __name__ == '__main__':
    finetune()
