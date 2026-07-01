from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VTraceOutput:
    vs: torch.Tensor
    policy_gradient_advantages: torch.Tensor
    rhos: torch.Tensor
    clipped_rhos: torch.Tensor
    clipped_pg_rhos: torch.Tensor
    cs: torch.Tensor


@dataclass
class VTraceLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    vs: torch.Tensor
    policy_gradient_advantages: torch.Tensor
    mean_rho: torch.Tensor
    max_rho: torch.Tensor


def from_importance_weights(
    behavior_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    discounts: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0,
    log_rho_clip: float = 20.0,
) -> VTraceOutput:
    """Compute IMPALA-style V-trace targets for fixed-length trajectories.

    Shapes are `[batch, time]` for log-probs, rewards, discounts and values,
    and `[batch]` for `bootstrap_value`. Targets are detached because they are
    used as regression / policy-gradient advantages for the learner policy.
    """

    if behavior_log_probs.shape != target_log_probs.shape:
        raise ValueError("behavior_log_probs and target_log_probs must have the same shape")
    if rewards.shape != values.shape or rewards.shape != target_log_probs.shape:
        raise ValueError("rewards, values and log-probs must all have shape [batch, time]")
    if discounts.shape != rewards.shape:
        raise ValueError("discounts must have shape [batch, time]")
    if bootstrap_value.shape != rewards[:, 0].shape:
        raise ValueError("bootstrap_value must have shape [batch]")

    with torch.no_grad():
        log_rhos = (target_log_probs - behavior_log_probs).clamp(min=-log_rho_clip, max=log_rho_clip)
        rhos = torch.exp(log_rhos)
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        cs = torch.clamp(rhos, max=clip_c_threshold)

        values_t_plus_1 = torch.cat([values[:, 1:], bootstrap_value[:, None]], dim=1)
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        reversed_vs_minus_v = []
        for t in range(rewards.shape[1] - 1, -1, -1):
            acc = deltas[:, t] + discounts[:, t] * cs[:, t] * acc
            reversed_vs_minus_v.append(acc)
        vs_minus_v = torch.stack(list(reversed(reversed_vs_minus_v)), dim=1)
        vs = values + vs_minus_v

        vs_t_plus_1 = torch.cat([vs[:, 1:], bootstrap_value[:, None]], dim=1)
        policy_gradient_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

    return VTraceOutput(
        vs=vs.detach(),
        policy_gradient_advantages=policy_gradient_advantages.detach(),
        rhos=rhos.detach(),
        clipped_rhos=clipped_rhos.detach(),
        clipped_pg_rhos=clipped_pg_rhos.detach(),
        cs=cs.detach(),
    )


def vtrace_actor_critic_loss(
    behavior_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    entropies: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float = 0.99,
    baseline_cost: float = 0.5,
    entropy_coeff: float = 0.004,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0,
) -> VTraceLossOutput:
    discounts = torch.full_like(rewards, float(gamma))
    vt = from_importance_weights(
        behavior_log_probs=behavior_log_probs,
        target_log_probs=target_log_probs,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        clip_c_threshold=clip_c_threshold,
    )
    policy_loss = -(target_log_probs * vt.policy_gradient_advantages).mean()
    value_loss = 0.5 * (vt.vs - values).pow(2).mean()
    entropy = entropies.mean()
    loss = policy_loss + baseline_cost * value_loss - entropy_coeff * entropy
    return VTraceLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        vs=vt.vs,
        policy_gradient_advantages=vt.policy_gradient_advantages,
        mean_rho=vt.rhos.mean(),
        max_rho=vt.rhos.max(),
    )
