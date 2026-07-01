from __future__ import annotations

import torch

from src.vtrace import from_importance_weights, vtrace_actor_critic_loss


def test_on_policy_vtrace_matches_discounted_returns_when_values_zero():
    rewards = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    discounts = torch.full_like(rewards, 0.5)
    values = torch.zeros_like(rewards)
    bootstrap = torch.zeros(1, dtype=torch.float32)
    behavior = torch.zeros_like(rewards)
    target = torch.zeros_like(rewards)

    out = from_importance_weights(
        behavior_log_probs=behavior,
        target_log_probs=target,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
    )

    expected = torch.tensor([[1.0 + 0.5 * 2.0 + 0.25 * 3.0, 2.0 + 0.5 * 3.0, 3.0]])
    torch.testing.assert_close(out.vs, expected)
    torch.testing.assert_close(out.policy_gradient_advantages, expected)
    torch.testing.assert_close(out.rhos, torch.ones_like(rewards))


def test_vtrace_clips_large_importance_ratios_and_returns_finite_loss():
    behavior = torch.zeros((2, 3), dtype=torch.float32)
    target = torch.full((2, 3), 10.0, dtype=torch.float32, requires_grad=True)
    values = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
    rewards = torch.ones((2, 3), dtype=torch.float32)
    entropies = torch.ones((2, 3), dtype=torch.float32)
    bootstrap = torch.zeros(2, dtype=torch.float32)

    loss_out = vtrace_actor_critic_loss(
        behavior_log_probs=behavior,
        target_log_probs=target,
        values=values,
        rewards=rewards,
        entropies=entropies,
        bootstrap_value=bootstrap,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
        clip_c_threshold=1.0,
    )

    assert torch.isfinite(loss_out.loss)
    assert float(loss_out.max_rho) > 1.0
    loss_out.loss.backward()
    assert target.grad is not None
    assert values.grad is not None

