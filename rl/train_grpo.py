"""GRPO training loop for PolyFormer using a Gaussian mixture policy."""
import argparse
import copy
from dataclasses import dataclass
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import PolicyWrapper, Trajectory
from .rewards import get_reward_fn


@dataclass
class Rollout:
    tokens: torch.LongTensor
    coords: torch.Tensor
    component_indices: torch.LongTensor
    log_probs: torch.Tensor
    encoder_out: Any


def run_rollout(policy: PolicyWrapper, encoder_out: Any, max_steps: int) -> Rollout:
    """Autoregressively sample a trajectory from ``policy``."""
    prev_tokens = torch.zeros((encoder_out.size(0), 0), dtype=torch.long, device=encoder_out.device)
    prev_coords = torch.zeros((encoder_out.size(0), 0, 2), device=encoder_out.device)
    tokens: List[torch.Tensor] = []
    coords: List[torch.Tensor] = []
    comp_ids: List[torch.Tensor] = []
    logps: List[torch.Tensor] = []
    for _ in range(max_steps):
        token, coord, comp, logp = policy.sample_step(encoder_out, prev_tokens, prev_coords)
        tokens.append(token)
        logps.append(logp)
        if coord is None:
            coord = torch.zeros((encoder_out.size(0), 2), device=encoder_out.device)
            comp = torch.zeros((encoder_out.size(0),), device=encoder_out.device, dtype=torch.long)
        coords.append(coord)
        comp_ids.append(comp)
        prev_tokens = torch.cat([prev_tokens, token.unsqueeze(1)], dim=1)
        prev_coords = torch.cat([prev_coords, coord.unsqueeze(1)], dim=1)
    tokens_tensor = torch.stack(tokens, dim=1)
    coords_tensor = torch.stack(coords, dim=1)
    comps_tensor = torch.stack(comp_ids, dim=1)
    logps_tensor = torch.stack(logps, dim=1).sum(dim=1)
    return Rollout(tokens_tensor, coords_tensor, comps_tensor, logps_tensor, encoder_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PolyFormer with GRPO")
    parser.add_argument("--reward-fn", type=str, default="length")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--total-iters", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-beta", type=float, default=0.1)
    parser.add_argument("--coord-token", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    # Placeholder model; a real implementation should load a pretrained PolyFormer
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocab_size = 10
            self.num_components = 5

        def forward(self, encoder_out=None, prev_tokens=None, prev_coords=None):
            bsz = prev_tokens.size(0)
            token_logits = torch.zeros(bsz, self.vocab_size, device=prev_tokens.device)
            mus = torch.zeros(bsz, self.num_components, 2, device=prev_tokens.device)
            log_sigmas = torch.zeros_like(mus)
            logits_pi = torch.zeros(bsz, self.num_components, device=prev_tokens.device)
            return token_logits, (mus, log_sigmas, logits_pi)

    base_model = DummyModel()
    policy = PolicyWrapper(base_model, coord_token=args.coord_token)
    reference = PolicyWrapper(copy.deepcopy(base_model), coord_token=args.coord_token)
    old_policy = PolicyWrapper(copy.deepcopy(base_model), coord_token=args.coord_token)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    reward_fn = get_reward_fn(args.reward_fn)

    for _ in range(args.total_iters):
        encoder_out = torch.zeros(args.batch_size * args.group_size, 8)
        rollout = run_rollout(old_policy, encoder_out, args.max_steps)
        # Compute rewards
        rewards = []
        for seq in rollout.tokens:
            rewards.append(reward_fn(seq.tolist()))
        rewards = torch.tensor(rewards, device=encoder_out.device)
        # Group-normalized advantages
        rewards_group = rewards.view(args.batch_size, args.group_size)
        r_mean = rewards_group.mean(dim=1, keepdim=True)
        r_std = rewards_group.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_group - r_mean) / r_std).view(-1)
        # Re-evaluate log-probabilities
        traj = Trajectory(
            tokens=rollout.tokens,
            coords=rollout.coords,
            component_indices=rollout.component_indices,
            encoder_out=rollout.encoder_out,
        )
        logp_new = policy.log_prob(traj)
        logp_ref = reference.log_prob(traj)
        logp_old = rollout.log_probs
        ratio = torch.exp(logp_new - logp_old)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantages
        surrogate = torch.min(unclipped, clipped)
        kl_estimate = (logp_new - logp_ref).mean()
        loss = -(surrogate.mean() - args.kl_beta * kl_estimate)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        old_policy.load_state_dict(policy.state_dict())

if __name__ == "__main__":
    main()
