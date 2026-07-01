from __future__ import annotations

import torch

from src.config import GameConfig
from src.train import joint_action_log_prob_and_entropy


def test_joint_action_logprob_sums_over_edges_not_mean():
    cfg = GameConfig()
    logits = torch.zeros((cfg.n_possible_edges, 2), dtype=torch.float32)
    dist = torch.distributions.Categorical(logits=logits)
    actions = torch.zeros(cfg.n_possible_edges, dtype=torch.long)
    log_prob, entropy = joint_action_log_prob_and_entropy(dist, actions)
    per_edge_log_prob = dist.log_prob(actions)
    assert torch.allclose(log_prob, per_edge_log_prob.sum())
    assert not torch.allclose(log_prob, per_edge_log_prob.mean())
    assert torch.allclose(entropy, dist.entropy().sum())

