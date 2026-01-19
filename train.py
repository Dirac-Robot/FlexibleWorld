"""
Training script for World Model
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from config import scope
from models import WorldModel
from models.world_model import create_world_model
from dataset import create_dataloader
from utils import save_checkpoint, save_video, make_grid
from utils.visualization import save_comparison


def compute_loss(outputs: dict, targets: torch.Tensor, config) -> dict:
    """Compute training loss."""
    recons = outputs['recons']
    post_logits = outputs['posterior_logits']
    prior_logits = outputs['prior_logits']

    # Reconstruction loss
    recon_loss = F.mse_loss(recons, targets)

    # KL divergence loss
    post_dist = torch.distributions.OneHotCategorical(logits=post_logits)
    prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)
    kl_loss = torch.distributions.kl_divergence(post_dist, prior_dist)
    kl_loss = kl_loss.sum(dim=-1).mean()  # sum over stoch_dim, mean over batch

    # Total loss
    total_loss = (
        config.train.loss.recon_weight*recon_loss +
        config.train.loss.kl_weight*kl_loss
    )

    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss,
    }


@scope
def train(config):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Create model
    model = create_world_model(config)
    model = model.to(device)
    logger.info(f'Created WorldModel with {sum(p.numel() for p in model.parameters()):,} parameters')

    # Create dataloader
    dataloader = create_dataloader(
        path=config.data.path,
        batch_size=config.train.batch_size,
        seq_len=config.train.seq_len,
        image_size=config.model.image_size,
        num_workers=config.data.num_workers,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
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

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.train.epochs}')
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
                    'kl': f'{losses["kl"].item():.4f}',
                })

            global_step += 1

        # Epoch summary
        n_batches = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        logger.info(f'Epoch {epoch+1}: loss={epoch_losses["total"]:.4f}, recon={epoch_losses["recon"]:.4f}, kl={epoch_losses["kl"]:.4f}')

        # Save checkpoint
        if (epoch + 1)%config.train.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                path=ckpt_dir/f'epoch_{epoch+1:04d}.pt',
                config=dict(config),
            )

        # Visualization
        if (epoch + 1)%config.train.save_every == 0:
            model.eval()
            with torch.no_grad():
                # Take first batch
                frames, actions = next(iter(dataloader))
                frames = frames[:4].to(device)
                actions = actions[:4].to(device)

                outputs = model(frames, actions)
                recons = outputs['recons']

                # Save comparison
                save_comparison(
                    frames[:, 0],
                    recons[:, 0],
                    ckpt_dir/f'recon_epoch_{epoch+1:04d}.png',
                )

                # Save imagined rollout
                state = outputs['final_state']
                imagined = model.imagine(state, actions)
                save_video(imagined[0], ckpt_dir/f'imagine_epoch_{epoch+1:04d}.gif')

    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.train.epochs,
        path=ckpt_dir/'final.pt',
        config=dict(config),
    )
    logger.info('Training complete!')


if __name__ == '__main__':
    train()
