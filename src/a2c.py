from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class A2CLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


def discounted_returns(rewards: Iterable[float], gamma: float, device: str | torch.device = "cpu") -> torch.Tensor:
    returns: List[float] = []
    running = 0.0
    for reward in reversed(list(rewards)):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)


def a2c_loss(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: Iterable[float],
    entropies: torch.Tensor,
    gamma: float = 0.99,
    baseline_cost: float = 0.5,
    entropy_coeff: float = 0.004,
) -> A2CLossOutput:
    returns = discounted_returns(rewards, gamma, device=values.device)
    if returns.numel() == 0:
        zero = values.sum() * 0.0
        return A2CLossOutput(zero, zero, zero, zero, returns, returns)
    advantages = returns - values
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = 0.5 * advantages.pow(2).mean()
    entropy = entropies.mean()
    loss = policy_loss + baseline_cost * value_loss - entropy_coeff * entropy
    return A2CLossOutput(loss, policy_loss, value_loss, entropy, returns, advantages)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    simulated_rounds: int,
    episodes: int,
    seed: int,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "simulated_rounds": simulated_rounds,
        "episodes": episodes,
        "seed": seed,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload

