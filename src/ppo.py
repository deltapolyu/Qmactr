from __future__ import annotations

from typing import List, Tuple

import torch


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(mask <= 0, -1e9)
    return torch.log_softmax(masked_logits, dim=-1)


def gae_advantages(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(rewards)
    adv = torch.zeros(n, dtype=torch.float32)
    returns = torch.zeros(n, dtype=torch.float32)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(n)):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        returns[t] = adv[t] + values[t]
        next_value = values[t]

    if bool(normalize):
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, returns


def ppo_clip_objective(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return torch.min(surr1, surr2).mean()
