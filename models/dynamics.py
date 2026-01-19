"""
RSSM Dynamics Model: Recurrent State-Space Model
Core of the world model - predicts next latent states given actions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class RSSMDynamics(nn.Module):
    """
    Recurrent State-Space Model (RSSM) for world modeling.

    State consists of:
    - Deterministic state (h): GRU hidden state, captures temporal context
    - Stochastic state (z): Categorical latent, captures uncertainty

    Two modes:
    - Prior (imagination): predicts z from h only
    - Posterior (observation): infers z from h and observation embedding
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        stoch_dim: int = 32,
        stoch_classes: int = 32,
        action_dim: int = 4,
        embed_dim: int = 256,
        discrete_actions: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.stoch_classes = stoch_classes
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.discrete_actions = discrete_actions

        # Action embedding
        if discrete_actions:
            self.action_embed = nn.Embedding(action_dim, hidden_dim)
        else:
            self.action_embed = nn.Linear(action_dim, hidden_dim)

        # Stochastic state size
        stoch_size = stoch_dim*stoch_classes

        # GRU for deterministic state
        self.gru = nn.GRUCell(stoch_size + hidden_dim, hidden_dim)

        # Prior network: h → z (for imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim*stoch_classes),
        )

        # Posterior network: h + embed → z (for observation)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim*stoch_classes),
        )

    def initial_state(self, batch_size: int, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """Create initial state (zeros)."""
        device = device or torch.device('cpu')
        return {
            'deter': torch.zeros(batch_size, self.hidden_dim, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device),
            'logits': torch.zeros(batch_size, self.stoch_dim, self.stoch_classes, device=device),
        }

    def imagine_step(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagination step: predict next state without observation.
        Used for planning/imagination rollouts.

        Args:
            state: current state dict
            action: (B,) discrete or (B, action_dim) continuous

        Returns:
            next_state: predicted next state dict
        """
        deter = state['deter']
        stoch = state['stoch'].reshape(deter.shape[0], -1)

        # Embed action
        action_emb = self.action_embed(action)

        # Update deterministic state
        gru_input = torch.cat([stoch, action_emb], dim=-1)
        next_deter = self.gru(gru_input, deter)

        # Sample stochastic state from prior
        prior_logits = self.prior_net(next_deter)
        prior_logits = prior_logits.reshape(-1, self.stoch_dim, self.stoch_classes)
        next_stoch = self._sample_stoch(prior_logits)

        return {
            'deter': next_deter,
            'stoch': next_stoch,
            'logits': prior_logits,
        }

    def observe_step(
        self,
        state: Dict[str, torch.Tensor],
        action: torch.Tensor,
        embed: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Observation step: update state with actual observation.
        Used during training.

        Args:
            state: current state dict
            action: (B,) discrete or (B, action_dim) continuous
            embed: (B, embed_dim) observation embedding

        Returns:
            next_state: posterior state (using observation)
            prior_state: prior state (for KL loss)
        """
        deter = state['deter']
        stoch = state['stoch'].reshape(deter.shape[0], -1)

        # Embed action
        action_emb = self.action_embed(action)

        # Update deterministic state
        gru_input = torch.cat([stoch, action_emb], dim=-1)
        next_deter = self.gru(gru_input, deter)

        # Prior (for KL loss)
        prior_logits = self.prior_net(next_deter)
        prior_logits = prior_logits.reshape(-1, self.stoch_dim, self.stoch_classes)

        # Posterior (using observation)
        post_input = torch.cat([next_deter, embed], dim=-1)
        post_logits = self.posterior_net(post_input)
        post_logits = post_logits.reshape(-1, self.stoch_dim, self.stoch_classes)
        post_stoch = self._sample_stoch(post_logits)

        posterior = {
            'deter': next_deter,
            'stoch': post_stoch,
            'logits': post_logits,
        }
        prior = {
            'deter': next_deter,
            'stoch': self._sample_stoch(prior_logits),
            'logits': prior_logits,
        }

        return posterior, prior

    def _sample_stoch(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution with straight-through gradient."""
        # Gumbel-softmax for differentiable sampling
        dist = torch.distributions.OneHotCategorical(logits=logits)
        sample = dist.sample()

        # Straight-through gradient
        probs = F.softmax(logits, dim=-1)
        sample = sample + probs - probs.detach()

        return sample

    def kl_loss(
        self,
        posterior_logits: torch.Tensor,
        prior_logits: torch.Tensor,
        free_nats: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.
        Uses free nats to prevent posterior collapse.
        """
        post_dist = torch.distributions.OneHotCategorical(logits=posterior_logits)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)

        kl = torch.distributions.kl_divergence(post_dist, prior_dist)
        kl = kl.sum(dim=-1)  # sum over stoch_dim
        kl = torch.clamp(kl, min=free_nats).mean()

        return kl

    def get_config(self) -> dict:
        return {
            'hidden_dim': self.hidden_dim,
            'stoch_dim': self.stoch_dim,
            'stoch_classes': self.stoch_classes,
            'action_dim': self.action_dim,
        }
