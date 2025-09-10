"""Policy wrapper around :class:`PolyFormerModel` for RL training."""
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from . import gmm_utils


@dataclass
class Trajectory:
    """Container storing rollout information.

    Attributes:
        tokens: Tensor of shape ``[B, T]`` containing generated token ids.
        coords: Tensor of shape ``[B, T, 2]`` containing coordinates for steps
            where a coordinate token was produced. Values are ignored otherwise.
        component_indices: Tensor of shape ``[B, T]`` with the index of the GMM
            component used to generate each coordinate. Ignored for non-coordinate
            steps.
        encoder_out: Cached encoder output required for re-evaluation.
    """

    tokens: torch.LongTensor
    coords: torch.Tensor
    component_indices: torch.LongTensor
    encoder_out: Any


class PolicyWrapper(nn.Module):
    """Thin wrapper exposing step-wise sampling and log-prob utilities."""

    def __init__(self, model: nn.Module, coord_token: Optional[int] = None):
        super().__init__()
        self.model = model
        self.coord_token = coord_token

    def forward(self, encoder_out, prev_tokens, prev_coords):
        """Forward pass through the underlying model.

        The wrapper assumes the wrapped model returns ``(token_logits, gmm_params)``
        where ``gmm_params`` is a tuple ``(mus, log_sigmas, logits_pi)``.
        """
        if hasattr(self.model, "decode_step"):
            token_logits, gmm_params = self.model.decode_step(
                encoder_out=encoder_out,
                prev_tokens=prev_tokens,
                prev_coords=prev_coords,
            )
        else:
            token_logits, gmm_params = self.model(
                encoder_out=encoder_out,
                prev_tokens=prev_tokens,
                prev_coords=prev_coords,
            )
        return token_logits, gmm_params

    def sample_step(self, encoder_out, prev_tokens, prev_coords):
        """Sample a token and, if necessary, a coordinate."""
        token_logits, gmm_params = self.forward(encoder_out, prev_tokens, prev_coords)
        token_dist = Categorical(logits=token_logits)
        token = token_dist.sample()
        logp_token = token_dist.log_prob(token)
        coord = None
        component_idx = None
        logp = logp_token
        if self.coord_token is not None and (token == self.coord_token).any():
            w, mu, sigma = gmm_utils.process_gmm_params(*gmm_params)
            coord, component_idx = gmm_utils.gmm_sample(w, mu, sigma)
            logp_coord = gmm_utils.gmm_log_prob(coord, w, mu, sigma, component_idx)
            logp = logp + logp_coord
        return token, coord, component_idx, logp

    def log_prob(self, trajectory: Trajectory) -> torch.Tensor:
        """Re-compute the joint log-probability of a trajectory."""
        tokens = trajectory.tokens
        coords = trajectory.coords
        comps = trajectory.component_indices
        encoder_out = trajectory.encoder_out
        batch_size, seq_len = tokens.shape
        prev_tokens = tokens.new_empty((batch_size, 0))
        prev_coords = coords.new_empty((batch_size, 0, coords.size(-1)))
        logps = []
        for t in range(seq_len):
            token_logits, gmm_params = self.forward(encoder_out, prev_tokens, prev_coords)
            token_t = tokens[:, t]
            token_dist = Categorical(logits=token_logits)
            logp_t = token_dist.log_prob(token_t)
            if self.coord_token is not None:
                mask = token_t == self.coord_token
                if mask.any():
                    w, mu, sigma = gmm_utils.process_gmm_params(*gmm_params)
                    coord_t = coords[:, t]
                    comp_t = comps[:, t]
                    logp_coord = gmm_utils.gmm_log_prob(coord_t, w, mu, sigma, comp_t)
                    logp_t = logp_t + logp_coord
                    prev_coords = torch.cat([prev_coords, coord_t.unsqueeze(1)], dim=1)
                else:
                    prev_coords = torch.cat(
                        [prev_coords, torch.zeros_like(prev_coords[:, :1])], dim=1
                    )
            logps.append(logp_t)
            prev_tokens = torch.cat([prev_tokens, token_t.unsqueeze(1)], dim=1)
        return torch.stack(logps, dim=1).sum(dim=1)
