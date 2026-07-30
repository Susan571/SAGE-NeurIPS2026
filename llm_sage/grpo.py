from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Callable


def group_relative_advantages(
    rewards: torch.Tensor, group_ids: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Normalize trajectory rewards within rollout groups (GRPO)."""
    if rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must have identical shapes")
    advantages = torch.empty_like(rewards, dtype=torch.float32)
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        group_rewards = rewards[mask].float()
        if len(group_rewards) == 1:
            advantages[mask] = 0.0
        else:
            advantages[mask] = (
                group_rewards - group_rewards.mean()
            ) / group_rewards.std(unbiased=False).clamp_min(epsilon)
    return advantages


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    beta_kl: float = 1e-3,
    clip_epsilon: float = 0.2,
    action_mask: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Clipped group-relative objective matching the paper's Equation (5)."""
    if new_logprobs.shape != old_logprobs.shape or new_logprobs.shape != reference_logprobs.shape:
        raise ValueError("new, old, and reference log-probabilities must match")
    if new_logprobs.ndim > 1:
        if advantages.ndim != 1 or advantages.shape[0] != new_logprobs.shape[0]:
            raise ValueError("batched step log-probabilities need one advantage per rollout")
        advantages = advantages[:, None].expand_as(new_logprobs)
    if action_mask is None:
        action_mask = torch.ones_like(new_logprobs, dtype=torch.bool)
    if action_mask.shape != new_logprobs.shape:
        raise ValueError("action_mask and log-probabilities must match")
    weights = action_mask.to(new_logprobs)
    if sample_weights is not None:
        if sample_weights.ndim == 1 and new_logprobs.ndim > 1:
            sample_weights = sample_weights[:, None].expand_as(new_logprobs)
        if sample_weights.shape != new_logprobs.shape:
            raise ValueError("sample_weights and log-probabilities must match")
        if torch.any(sample_weights < 0):
            raise ValueError("sample_weights must be nonnegative")
        weights = weights * sample_weights.to(new_logprobs)
    denominator = weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    log_ratio = new_logprobs - old_logprobs
    ratio = log_ratio.clamp(-20.0, 20.0).exp()
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    policy_loss = -(surrogate * weights).sum() / denominator

    # Monte Carlo estimate of log(pi_theta / pi_ref), exactly as written in
    # Equation (5), rather than a different positive-only KL estimator.
    kl = ((new_logprobs - reference_logprobs) * weights).sum() / denominator
    loss = policy_loss + beta_kl * kl
    return loss, {"policy_loss": policy_loss.detach(), "kl": kl.detach()}


@dataclass
class GRPOBatch:
    """One or more prompt groups prepared by a rollout worker."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor
    step_ids: torch.Tensor
    old_logprobs: torch.Tensor
    reference_logprobs: torch.Tensor
    action_mask: torch.Tensor
    rewards: torch.Tensor
    group_ids: torch.Tensor
    rollout_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "GRPOBatch":
        return GRPOBatch(
            **{
                name: getattr(self, name).to(device)
                for name in self.__dataclass_fields__
            }
        )


def step_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    step_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum token log-probabilities within each coarse reasoning action."""
    shifted_logits = logits[:, :-1].log_softmax(-1)
    targets = input_ids[:, 1:]
    token_logprobs = shifted_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask[:, 1:].to(token_logprobs)
    token_step_ids = step_ids[:, 1:] * completion_mask[:, 1:]
    max_steps = max(int(token_step_ids.max()), 1)
    result = torch.zeros(
        input_ids.shape[0],
        max_steps,
        device=token_logprobs.device,
        dtype=token_logprobs.dtype,
    )
    counts = torch.zeros_like(result)
    indices = (token_step_ids - 1).clamp_min(0)
    result.scatter_add_(1, indices, token_logprobs * mask)
    counts.scatter_add_(1, indices, mask)
    return result, counts > 0


def sequence_logprobs(
    logits: torch.Tensor, input_ids: torch.Tensor, completion_mask: torch.Tensor
) -> torch.Tensor:
    """Mean completion log-probability retained for external compatibility."""
    step_ids = completion_mask.long()
    per_step, valid = step_logprobs(logits, input_ids, completion_mask, step_ids)
    return per_step.sum(-1) / valid.sum(-1).clamp_min(1)


def grpo_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: GRPOBatch,
    beta_kl: float = 1e-3,
    clip_epsilon: float = 0.2,
    max_grad_norm: float = 1.0,
    accumulation_steps: int = 1,
    accumulation_index: int = 0,
    force_step: bool = False,
) -> dict[str, float]:
    """Run one end-to-end group-relative policy update.

    The reference and old-policy log-probabilities must be captured by rollout
    workers before the update. Structural scores enter indirectly through
    ``batch.rewards`` after Equation (E.2)'s terminal-plus-mean-potential
    construction. They also affect guided rollout sampling.
    """
    rollout_advantages = group_relative_advantages(batch.rewards, batch.group_ids).to(
        batch.input_ids.device
    )
    if batch.rollout_ids.numel() == 0 or int(batch.rollout_ids.min()) < 0:
        raise ValueError("rollout_ids must contain nonnegative rollout indices")
    if int(batch.rollout_ids.max()) >= len(batch.rewards):
        raise ValueError("rollout_ids index beyond the reward vector")
    advantages = rollout_advantages[batch.rollout_ids]
    actions_per_rollout = torch.bincount(
        batch.rollout_ids, minlength=len(batch.rewards)
    ).to(batch.rewards)
    if torch.any(actions_per_rollout == 0):
        raise ValueError("every rewarded rollout must contain at least one action")
    rollouts_per_group = torch.empty_like(batch.rewards, dtype=torch.float32)
    for group_id in torch.unique(batch.group_ids):
        group_mask = batch.group_ids == group_id
        rollouts_per_group[group_mask] = group_mask.sum()
    # Equation (5) averages 1/T_i within each rollout and 1/G within
    # each prompt group. This avoids overweighting longer trajectories.
    sample_weights = (
        actions_per_rollout[batch.rollout_ids].reciprocal()
        * rollouts_per_group[batch.rollout_ids].reciprocal()
    )
    output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        use_cache=False,
    )
    new_logprobs, action_mask = step_logprobs(
        output.logits,
        batch.input_ids,
        batch.completion_mask,
        batch.step_ids,
    )
    loss, metrics = grpo_loss(
        new_logprobs,
        batch.old_logprobs,
        batch.reference_logprobs,
        advantages,
        beta_kl,
        clip_epsilon,
        batch.action_mask & action_mask,
        sample_weights,
    )
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    if accumulation_index == 0:
        optimizer.zero_grad(set_to_none=True)
    (loss / accumulation_steps).backward()
    should_step = force_step or accumulation_index + 1 == accumulation_steps
    grad_norm = torch.tensor(float("nan"))
    if should_step:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(metrics["policy_loss"]),
        "kl": float(metrics["kl"]),
        "grad_norm": float(grad_norm),
        "reward_mean": float(batch.rewards.float().mean()),
        "optimizer_step": float(should_step),
    }
